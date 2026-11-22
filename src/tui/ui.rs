//! Draws the interactive UI: the search field, the results grid or detail view, and the footer.

use ratatui::Frame;
use ratatui::buffer::Buffer;
use ratatui::layout::{Alignment, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Clear, Paragraph, Widget, Wrap};

use super::app::{App, Focus, View, file_name};
use super::images::{Fit, ImageKey, ImageState, Images, Variant};
use super::kitty;

const CARD_MIN_WIDTH: u16 = 24;
const CARD_HEIGHT: u16 = 16;
const GUTTER: u16 = 1;
const THUMBNAIL_ROWS: u16 = 13;
const FILMSTRIP_HEIGHT: u16 = 9;
const FILMSTRIP_SLOT_WIDTH: u16 = 16;

const ACCENT: Color = Color::Blue;
const BLURRED: Color = Color::DarkGray;

fn muted() -> Style {
  Style::default().add_modifier(Modifier::DIM)
}

/// Where things were drawn, for mapping mouse clicks back to them.
#[derive(Default)]
pub struct HitMap {
  pub input: Rect,
  pub grid: Rect,
  pub cards: Vec<(Rect, usize)>,
  pub detail_image: Rect,
  pub filmstrip: Vec<(Rect, isize)>,
  /// The grid's full height in lines, for scrolling.
  pub grid_lines: usize,
}

pub struct Context<'a> {
  pub images: &'a mut Images,
  pub cell_pixels: (u16, u16),
}

pub fn draw(frame: &mut Frame<'_>, app: &mut App, context: &mut Context<'_>) -> HitMap {
  let area = frame.area();
  let mut hits = HitMap::default();
  if area.height < 8 || area.width < 10 {
    return hits;
  }
  let input = Rect::new(area.x + 1, area.y + 1, area.width - 2, 3);
  let labels = Rect::new(area.x + 1, area.y + 4, area.width - 2, 1);
  let footer_height = 2;
  let main = Rect::new(area.x, area.y + 5, area.width, area.height - 5 - footer_height);
  let path_line = Rect::new(area.x + 1, main.bottom(), area.width - 2, 1);
  let hotkeys = Rect::new(area.x + 1, main.bottom() + 1, area.width - 2, 1);

  hits.input = input;
  draw_input(frame, app, input);
  draw_labels(frame.buffer_mut(), app, labels);
  match app.view {
    View::Grid => draw_grid(frame.buffer_mut(), app, context, main, &mut hits),
    View::Detail => draw_detail(frame.buffer_mut(), app, context, main, &mut hits),
  }
  draw_footer(frame.buffer_mut(), app, path_line, hotkeys);
  draw_notifications(frame.buffer_mut(), app, Rect::new(area.x, area.y, area.width, main.bottom()));
  hits
}

fn border(focused: bool) -> (BorderType, Style) {
  if focused {
    (BorderType::Thick, Style::default().fg(ACCENT))
  } else {
    (BorderType::Plain, Style::default().fg(BLURRED))
  }
}

fn draw_input(frame: &mut Frame<'_>, app: &App, area: Rect) {
  let focused = app.focus == Focus::Search;
  let (border_type, border_style) = border(focused);
  let mut block = Block::default().borders(Borders::ALL).border_type(border_type).border_style(border_style);
  if let Some(status) = app.status {
    block = block.title(Span::styled(status, Style::default().fg(Color::Reset)));
  }
  let inner = block.inner(area);
  block.render(area, frame.buffer_mut());
  let width = usize::from(inner.width.saturating_sub(1)).max(1);
  if app.input.text.is_empty() {
    Paragraph::new(format!("Search images in {}…", app.directory)).style(muted()).render(inner, frame.buffer_mut());
  } else {
    // Scroll horizontally to keep the cursor visible.
    let chars: Vec<char> = app.input.text.chars().collect();
    let start = app.input.cursor.saturating_sub(width);
    let visible: String = chars[start..].iter().take(width + 1).collect();
    Paragraph::new(visible).render(inner, frame.buffer_mut());
  }
  if focused {
    let column = app.input.cursor.min(width) as u16;
    frame.set_cursor_position((inner.x + column, inner.y));
  }
}

fn draw_labels(buffer: &mut Buffer, app: &App, area: Rect) {
  let order = if app.ranked { "Search results · Most similar first" } else { "All images · Recently modified first" };
  let order_width = (order.chars().count() as u16).min(area.width / 2);
  let scope_area = Rect::new(area.x, area.y, area.width.saturating_sub(order_width + 2), 1);
  Paragraph::new(ellipsize(&format!("{} · Including subfolders", app.directory), usize::from(scope_area.width)))
    .render(scope_area, buffer);
  let order_area = Rect::new(area.right() - order_width, area.y, order_width, 1);
  Paragraph::new(ellipsize(order, usize::from(order_width))).alignment(Alignment::Right).render(order_area, buffer);
}

fn ellipsize(text: &str, width: usize) -> String {
  if text.chars().count() <= width {
    return text.to_string();
  }
  let mut shortened: String = text.chars().take(width.saturating_sub(1)).collect();
  shortened.push('…');
  shortened
}

/// Column widths of a grid that fits as many columns of at least the minimum width as possible.
fn grid_columns(width: u16) -> Vec<u16> {
  let count = ((width + GUTTER) / (CARD_MIN_WIDTH + GUTTER)).max(1);
  let available = width.saturating_sub((count - 1) * GUTTER);
  (0..count).map(|index| available / count + u16::from(index < available % count)).collect()
}

fn draw_grid(buffer: &mut Buffer, app: &mut App, context: &mut Context<'_>, area: Rect, hits: &mut HitMap) {
  let grid = Rect::new(area.x + 1, area.y, area.width.saturating_sub(3), area.height);
  hits.grid = grid;
  let widths = grid_columns(grid.width);
  let columns = widths.len();
  let rows = app.results.len().div_ceil(columns);
  let lines = (rows * usize::from(CARD_HEIGHT + GUTTER)).saturating_sub(usize::from(GUTTER));
  let viewport = usize::from(grid.height);
  app.grid.columns = columns;
  app.grid.viewport_rows = viewport;
  hits.grid_lines = lines;

  if app.reveal_selection && !app.results.is_empty() {
    app.reveal_selection = false;
    let top = app.selected / columns * usize::from(CARD_HEIGHT + GUTTER);
    let bottom = top + usize::from(CARD_HEIGHT);
    if top < app.scroll {
      app.scroll = top;
    } else if bottom > app.scroll + viewport {
      app.scroll = bottom.saturating_sub(viewport);
    }
  }
  app.scroll = app.scroll.min(lines.saturating_sub(viewport));

  let first_row = app.scroll / usize::from(CARD_HEIGHT + GUTTER);
  for row in first_row..rows {
    let top = row * usize::from(CARD_HEIGHT + GUTTER);
    if top >= app.scroll + viewport {
      break;
    }
    let mut x = grid.x;
    for (column, &width) in widths.iter().enumerate() {
      let index = row * columns + column;
      if index >= app.results.len() {
        break;
      }
      let y = i32::from(grid.y) + top as i32 - app.scroll as i32;
      draw_card(buffer, app, context, index, (x, y), width, grid, hits);
      x += width + GUTTER;
    }
  }
  draw_scrollbar(buffer, Rect::new(grid.right() + 1, grid.y, 1, grid.height), app.scroll, lines, viewport);
}

#[allow(clippy::too_many_arguments)]
fn draw_card(
  buffer: &mut Buffer,
  app: &App,
  context: &mut Context<'_>,
  index: usize,
  (x, y): (u16, i32),
  width: u16,
  clip: Rect,
  hits: &mut HitMap,
) {
  let result = &app.results[index];
  let thumbnail_cells = (width.saturating_sub(4), THUMBNAIL_ROWS);
  let key = ImageKey {
    path: result.filepath.clone(),
    variant: Variant::Preview,
    fit: Fit::Natural,
    cells: thumbnail_cells,
    cell_pixels: context.cell_pixels,
  };
  let state = context.images.get(&key);
  let failed = matches!(state, ImageState::Failed(_));
  let ready = match state {
    ImageState::Ready { id, cells, offset } => Some((id, cells, offset)),
    _ => None,
  };

  // Render the card off-screen, then copy its visible rows, so cards clip at the grid's edges.
  let mut card = Buffer::empty(Rect::new(0, 0, width, CARD_HEIGHT));
  let selected = index == app.selected && app.focus == Focus::Results;
  let (border_type, mut border_style) = border(selected);
  if failed {
    border_style = Style::default().fg(Color::Red);
  }
  Block::default()
    .borders(Borders::ALL)
    .border_type(border_type)
    .border_style(border_style)
    .render(card.area, &mut card);
  let label = match result.score {
    Some(score) => format!("{score:.3}  {}", file_name(&result.filepath)),
    None => file_name(&result.filepath).to_string(),
  };
  let label_area = Rect::new(2, CARD_HEIGHT - 2, width.saturating_sub(4), 1);
  Paragraph::new(ellipsize(&label, usize::from(label_area.width))).style(muted()).render(label_area, &mut card);
  for row in 0..CARD_HEIGHT {
    let target_y = y + i32::from(row);
    if target_y < i32::from(clip.top()) || target_y >= i32::from(clip.bottom()) {
      continue;
    }
    for column in 0..width {
      if let Some(cell) = buffer.cell_mut((x + column, target_y as u16)) {
        *cell = card[(column, row)].clone();
      }
    }
  }
  if let Some((id, cells, offset)) = ready {
    let origin = (i32::from(x + 2 + offset.0), y + 1 + i32::from(offset.1));
    kitty::draw(buffer, id, origin, cells, clip);
  }

  let visible_top = y.max(i32::from(clip.top()));
  let visible_bottom = (y + i32::from(CARD_HEIGHT)).min(i32::from(clip.bottom()));
  if visible_bottom > visible_top {
    hits.cards.push((Rect::new(x, visible_top as u16, width, (visible_bottom - visible_top) as u16), index));
  }
}

fn draw_scrollbar(buffer: &mut Buffer, area: Rect, scroll: usize, lines: usize, viewport: usize) {
  if lines <= viewport || area.height == 0 {
    return;
  }
  let height = usize::from(area.height);
  let thumb = (height * viewport / lines).max(1);
  let start = (height - thumb) * scroll / (lines - viewport);
  for row in 0..height {
    let symbol = if (start..start + thumb).contains(&row) { "┃" } else { "│" };
    let style = if (start..start + thumb).contains(&row) { Style::default() } else { Style::default().fg(BLURRED) };
    buffer.set_string(area.x, area.y + row as u16, symbol, style);
  }
}

fn draw_detail(buffer: &mut Buffer, app: &mut App, context: &mut Context<'_>, area: Rect, hits: &mut HitMap) {
  let filmstrip_height = FILMSTRIP_HEIGHT.min(area.height / 2);
  let frame = Rect::new(area.x, area.y, area.width, area.height - filmstrip_height);
  let image_box = Rect::new(frame.x + 1, frame.y + 1, frame.width.saturating_sub(2), frame.height.saturating_sub(2));
  hits.detail_image = frame;
  let neighbors = usize::from((area.width / FILMSTRIP_SLOT_WIDTH).saturating_sub(1) / 2);
  app.filmstrip_neighbors = neighbors;

  let detail_key = |path: &str| ImageKey {
    path: path.to_string(),
    variant: Variant::Detail,
    fit: Fit::Natural,
    cells: (image_box.width, image_box.height),
    cell_pixels: context.cell_pixels,
  };
  let status = match app.selected_result() {
    None => Some("No results".to_string()),
    Some(result) => match context.images.get(&detail_key(&result.filepath)) {
      ImageState::Ready { id, cells, offset } => {
        let origin = (i32::from(image_box.x + offset.0), i32::from(image_box.y + offset.1));
        kitty::draw(buffer, id, origin, cells, image_box);
        None
      }
      ImageState::Loading => Some("Loading higher-resolution image…".to_string()),
      ImageState::Failed(error) => Some(format!("Unable to load image: {error}")),
    },
  };
  if let Some(status) = status {
    let line = Rect::new(frame.x + 1, frame.y + frame.height / 2, frame.width.saturating_sub(2), 1);
    Paragraph::new(ellipsize(&status, usize::from(line.width)))
      .style(muted())
      .alignment(Alignment::Center)
      .render(line, buffer);
  }
  // Load the neighbors too, so that browsing shows them right away.
  for neighbor in [app.selected.checked_sub(1), Some(app.selected + 1)].into_iter().flatten() {
    if let Some(result) = app.results.get(neighbor) {
      let _ = context.images.get(&detail_key(&result.filepath));
    }
  }

  let strip = Rect::new(area.x, frame.bottom(), area.width, filmstrip_height);
  let slots = 2 * neighbors + 1;
  let slot_width = |slot: usize| strip.width / slots as u16 + u16::from(slot < usize::from(strip.width) % slots);
  let thumbnail_frame = |slot_area: Rect, selected: bool| {
    if selected {
      return slot_area;
    }
    let frame_width = (slot_area.width * 9 / 10).max(1);
    let frame_height = 7.min(slot_area.height);
    Rect::new(
      slot_area.x + (slot_area.width - frame_width) / 2,
      slot_area.y + (slot_area.height - frame_height) / 2,
      frame_width,
      frame_height,
    )
  };
  let thumbnail_key = |path: &str, frame: Rect| ImageKey {
    path: path.to_string(),
    variant: Variant::Preview,
    fit: Fit::Fill,
    cells: (frame.width.saturating_sub(4), frame.height.saturating_sub(2)),
    cell_pixels: context.cell_pixels,
  };
  let mut x = strip.x;
  for slot in 0..slots {
    let offset = slot as isize - neighbors as isize;
    let slot_area = Rect::new(x, strip.y, slot_width(slot), strip.height);
    x += slot_area.width;
    let Some(result) = app.selected.checked_add_signed(offset).and_then(|index| app.results.get(index)) else {
      continue;
    };
    let selected = offset == 0;
    let frame = thumbnail_frame(slot_area, selected);
    let (border_type, border_style) = border(selected);
    Block::default().borders(Borders::ALL).border_type(border_type).border_style(border_style).render(frame, buffer);
    let key = thumbnail_key(&result.filepath, frame);
    let image_area = Rect::new(frame.x + 2, frame.y + 1, key.cells.0, key.cells.1);
    if image_area.width > 0
      && image_area.height > 0
      && let ImageState::Ready { id, cells, offset } = context.images.get(&key)
    {
      let origin = (i32::from(image_area.x + offset.0), i32::from(image_area.y + offset.1));
      kitty::draw(buffer, id, origin, cells, image_area);
    }
    hits.filmstrip.push((slot_area, offset));
  }
  // Load the previews hidden past each edge at the size of the slot browsing moves them into, so that it
  // reveals them without a flicker.
  for (hidden, slot) in [(app.selected.checked_sub(neighbors + 1), 0), (Some(app.selected + neighbors + 1), slots - 1)]
  {
    let frame = thumbnail_frame(Rect::new(strip.x, strip.y, slot_width(slot), strip.height), neighbors == 0);
    if let Some(result) = hidden.and_then(|index| app.results.get(index))
      && frame.width > 4
      && frame.height > 2
    {
      let _ = context.images.get(&thumbnail_key(&result.filepath, frame));
    }
  }
}

fn draw_footer(buffer: &mut Buffer, app: &App, path_line: Rect, hotkeys: Rect) {
  if app.show_selected_path
    && let Some(result) = app.selected_result()
  {
    let label = match result.score {
      Some(score) => format!("{score:.3}  {}", result.filepath),
      None => result.filepath.clone(),
    };
    Paragraph::new(ellipsize(&label, usize::from(path_line.width)))
      .style(muted())
      .alignment(Alignment::Center)
      .render(path_line, buffer);
  }
  let highlight = Style::default().bg(BLURRED).fg(Color::White);
  let mut spans = Vec::new();
  for (index, (key, label)) in app.hotkeys().into_iter().enumerate() {
    if index > 0 {
      spans.push(Span::raw("   "));
    }
    spans.push(Span::styled(format!(" {key} "), highlight));
    spans.push(Span::raw(format!(" {label}")));
  }
  Paragraph::new(Line::from(spans)).alignment(Alignment::Center).render(hotkeys, buffer);
}

fn draw_notifications(buffer: &mut Buffer, app: &App, area: Rect) {
  let width = 50.min(area.width.saturating_sub(4));
  let mut bottom = area.bottom();
  for notification in app.notifications.iter().rev() {
    let text_width = usize::from(width.saturating_sub(4)).max(1);
    let message_lines = notification.message.chars().count().div_ceil(text_width).max(1) as u16;
    let height = message_lines + 3;
    if bottom < area.y + height {
      break;
    }
    let toast = Rect::new(area.right() - width - 2, bottom - height, width, height);
    bottom -= height;
    let color = if notification.error { Color::Red } else { ACCENT };
    Clear.render(toast, buffer);
    let block =
      Block::default().borders(Borders::ALL).border_type(BorderType::Thick).border_style(Style::default().fg(color));
    let text = vec![
      Line::styled(notification.title.clone(), Style::default().add_modifier(Modifier::BOLD)),
      Line::raw(notification.message.clone()),
    ];
    Paragraph::new(text)
      .wrap(Wrap { trim: false })
      .block(block.padding(ratatui::widgets::Padding::horizontal(1)))
      .render(toast, buffer);
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn fits_as_many_columns_as_possible() {
    assert_eq!(grid_columns(10), [10]);
    assert_eq!(grid_columns(24), [24]);
    assert_eq!(grid_columns(49), [24, 24]);
    assert_eq!(grid_columns(80), [26, 26, 26]);
    assert_eq!(grid_columns(81), [27, 26, 26]);
  }

  #[test]
  fn ellipsizes_long_text() {
    assert_eq!(ellipsize("abcdef", 4), "abc…");
    assert_eq!(ellipsize("abc", 4), "abc");
  }
}
