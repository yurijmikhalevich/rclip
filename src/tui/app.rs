//! The interactive UI's state and behavior, independent of the terminal.
//!
//! Input events update the state and return [`Effect`]s (searches to start, paths to copy, ...)
//! for the event loop to carry out, which keeps this module free of I/O and easy to test.

use std::time::{Duration, Instant};

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use super::worker::{Request, Response, TuiResult};
use crate::db::ImageCursor;

/// Searches start this long after the last keystroke.
pub const DEBOUNCE: Duration = Duration::from_millis(250);
const NOTIFICATION_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Focus {
  Search,
  Results,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum View {
  Grid,
  Detail,
}

#[derive(Debug, PartialEq)]
pub enum Effect {
  Request(u64, Request),
  CopyPath(String),
  CopyImage(String),
  Download(String),
  Quit,
}

#[derive(Debug)]
pub struct Notification {
  pub title: String,
  pub message: String,
  pub error: bool,
  pub expires: Instant,
}

/// A single-line text field.
#[derive(Debug, Default)]
pub struct Input {
  pub text: String,
  /// The cursor position, in characters.
  pub cursor: usize,
}

impl Input {
  fn byte_index(&self, chars: usize) -> usize {
    self.text.char_indices().nth(chars).map_or(self.text.len(), |(index, _)| index)
  }

  fn len(&self) -> usize {
    self.text.chars().count()
  }

  /// Applies an editing key; returns whether the text changed, or `None` if the key is not an
  /// editing key.
  fn edit(&mut self, key: &KeyEvent) -> Option<bool> {
    let control = key.modifiers.contains(KeyModifiers::CONTROL);
    let before = self.text.clone();
    match key.code {
      KeyCode::Char('a') if control => self.cursor = 0,
      KeyCode::Char('e') if control => self.cursor = self.len(),
      KeyCode::Char('u') if control => {
        let end = self.byte_index(self.cursor);
        self.text.replace_range(..end, "");
        self.cursor = 0;
      }
      KeyCode::Char('k') if control => {
        let start = self.byte_index(self.cursor);
        self.text.truncate(start);
      }
      KeyCode::Char('w') if control => {
        let end = self.byte_index(self.cursor);
        let trimmed = self.text[..end].trim_end();
        let start = trimmed.rfind(char::is_whitespace).map_or(0, |index| index + 1);
        self.text.replace_range(start..end, "");
        self.cursor = self.text[..start].chars().count();
      }
      KeyCode::Char(c) if !control && !key.modifiers.contains(KeyModifiers::ALT) => {
        let index = self.byte_index(self.cursor);
        self.text.insert(index, c);
        self.cursor += 1;
      }
      KeyCode::Backspace if self.cursor > 0 => {
        let start = self.byte_index(self.cursor - 1);
        let end = self.byte_index(self.cursor);
        self.text.replace_range(start..end, "");
        self.cursor -= 1;
      }
      KeyCode::Delete if self.cursor < self.len() => {
        let start = self.byte_index(self.cursor);
        let end = self.byte_index(self.cursor + 1);
        self.text.replace_range(start..end, "");
      }
      KeyCode::Left => self.cursor = self.cursor.saturating_sub(1),
      KeyCode::Right => self.cursor = (self.cursor + 1).min(self.len()),
      KeyCode::Home => self.cursor = 0,
      KeyCode::End => self.cursor = self.len(),
      KeyCode::Backspace | KeyCode::Delete => {}
      _ => return None,
    }
    Some(self.text != before)
  }

  pub fn query(&self) -> &str {
    self.text.trim()
  }
}

/// Grid geometry from the last frame, used to move between rows and to scroll.
#[derive(Clone, Copy, Debug, Default)]
pub struct GridGeometry {
  pub columns: usize,
  pub viewport_rows: usize,
}

pub struct App {
  /// The searched directory, with the home directory abbreviated to `~`.
  pub directory: String,
  pub input: Input,
  pub focus: Focus,
  pub view: View,
  pub results: Vec<TuiResult>,
  pub selected: usize,
  /// Whether the footer shows the selected image's path.
  pub show_selected_path: bool,
  /// The input's border title: "Searching…", "Loading…", or "No results".
  pub status: Option<&'static str>,
  /// Whether the results are ranked search results rather than browsed images.
  pub ranked: bool,
  pub notifications: Vec<Notification>,
  /// The first visible line of the grid.
  pub scroll: usize,
  /// Set when the selection moved and the grid should scroll it into view.
  pub reveal_selection: bool,
  pub grid: GridGeometry,
  /// Thumbnails shown on each side of the selection in the detail view's filmstrip.
  pub filmstrip_neighbors: usize,
  remaining: Vec<TuiResult>,
  next_cursor: Option<ImageCursor>,
  generation: u64,
  loading_more: bool,
  pending_detail_advance: bool,
  batch_size: usize,
  debounce: Option<Instant>,
}

impl App {
  /// Creates the app and starts browsing all images, as the Python version does on mount.
  pub fn new(directory: String, batch_size: usize) -> (Self, Vec<Effect>) {
    let mut app = App {
      directory,
      input: Input::default(),
      focus: Focus::Search,
      view: View::Grid,
      results: Vec::new(),
      selected: 0,
      show_selected_path: false,
      status: None,
      ranked: false,
      notifications: Vec::new(),
      scroll: 0,
      reveal_selection: false,
      grid: GridGeometry { columns: 1, viewport_rows: 1 },
      filmstrip_neighbors: 0,
      remaining: Vec::new(),
      next_cursor: None,
      generation: 0,
      loading_more: false,
      pending_detail_advance: false,
      batch_size,
      debounce: None,
    };
    let effects = app.begin_search(String::new());
    (app, effects)
  }

  pub fn selected_result(&self) -> Option<&TuiResult> {
    self.results.get(self.selected)
  }

  /// When the next timer-driven update is due.
  pub fn next_deadline(&self) -> Option<Instant> {
    self.notifications.iter().map(|notification| notification.expires).chain(self.debounce).min()
  }

  /// Fires the debounced search and expires notifications.
  pub fn tick(&mut self, now: Instant) -> Vec<Effect> {
    self.notifications.retain(|notification| notification.expires > now);
    if self.debounce.is_some_and(|deadline| deadline <= now) {
      self.debounce = None;
      return self.begin_search(self.input.query().to_string());
    }
    Vec::new()
  }

  pub fn notify(&mut self, title: impl Into<String>, message: impl Into<String>, error: bool) {
    self.notifications.push(Notification {
      title: title.into(),
      message: message.into(),
      error,
      expires: Instant::now() + NOTIFICATION_TIMEOUT,
    });
  }

  fn begin_search(&mut self, query: String) -> Vec<Effect> {
    self.generation += 1;
    self.pending_detail_advance = false;
    self.loading_more = false;
    self.next_cursor = None;
    self.remaining.clear();
    self.status = Some("Searching…");
    let request = if query.is_empty() { Request::Browse(None) } else { Request::Search(query) };
    vec![Effect::Request(self.generation, request)]
  }

  fn is_current(&self, generation: u64, query: &str) -> bool {
    generation == self.generation && self.input.query() == query
  }

  /// Applies a finished search or browse request.
  pub fn on_response(&mut self, response: Response) -> Vec<Effect> {
    let (query, append) = match &response.request {
      Request::Search(query) => (query.as_str(), false),
      Request::Browse(after) => ("", after.is_some()),
    };
    if !self.is_current(response.generation, query) {
      return Vec::new();
    }
    match response.outcome {
      Ok((results, next_cursor)) => {
        let has_results = !results.is_empty();
        let ranked = !query.is_empty();
        self.show_results(results, next_cursor, append, ranked);
        if !append && !has_results {
          self.status = Some("No results");
        }
        Vec::new()
      }
      Err(message) => {
        self.loading_more = false;
        self.pending_detail_advance = false;
        self.status = None;
        let title = if append { "Unable to load more images" } else { "Search failed" };
        self.notify(title, message, true);
        Vec::new()
      }
    }
  }

  fn show_results(
    &mut self,
    mut results: Vec<TuiResult>,
    next_cursor: Option<ImageCursor>,
    append: bool,
    ranked: bool,
  ) {
    if !append {
      self.show_selected_path = false;
      self.results.clear();
      if ranked && results.len() > self.batch_size {
        self.remaining = results.split_off(self.batch_size);
      }
    }
    let appended = !results.is_empty();
    self.results.extend(results);
    self.next_cursor = next_cursor;
    self.loading_more = false;
    self.ranked = ranked;
    self.status = None;
    if append && self.pending_detail_advance {
      self.pending_detail_advance = false;
      if appended && self.view == View::Detail {
        self.move_detail(1);
      }
    }
    if !append {
      self.scroll = 0;
      self.selected = 0;
    }
    if self.view == View::Detail {
      self.show_selected_path = true;
    }
  }

  /// Shows more results when the user nears the end of the loaded ones.
  fn load_more(&mut self) -> Vec<Effect> {
    if self.loading_more
      || (self.view == View::Detail
        && !self.pending_detail_advance
        && self.selected + self.filmstrip_neighbors + 1 < self.results.len())
    {
      return Vec::new();
    }
    if !self.input.query().is_empty() {
      if self.remaining.is_empty() || !self.ranked {
        return Vec::new();
      }
      let count = self.batch_size.min(self.remaining.len());
      let batch: Vec<TuiResult> = self.remaining.drain(..count).collect();
      self.show_results(batch, None, true, true);
      return Vec::new();
    }
    let Some(cursor) = self.next_cursor.clone() else { return Vec::new() };
    self.loading_more = true;
    self.status = Some("Loading…");
    vec![Effect::Request(self.generation, Request::Browse(Some(cursor)))]
  }

  /// Called after each frame: loads more results once the grid is scrolled near its end, or the
  /// detail view nears the last loaded result.
  pub fn after_frame(&mut self, grid_lines: usize) -> Vec<Effect> {
    match self.view {
      View::Grid => {
        let max_scroll = grid_lines.saturating_sub(self.grid.viewport_rows);
        if max_scroll.saturating_sub(self.scroll) <= self.grid.viewport_rows && !self.results.is_empty() {
          return self.load_more();
        }
        Vec::new()
      }
      View::Detail => self.load_more(),
    }
  }

  fn can_act_on_selection(&self) -> bool {
    self.selected_result().is_some() && (self.view == View::Detail || self.focus == Focus::Results)
  }

  pub fn hotkeys(&self) -> Vec<(String, &'static str)> {
    let search_focused = self.focus == Focus::Search;
    let mut keys: Vec<(String, &'static str)> =
      vec![if search_focused { ("Down".into(), "Browse") } else { ("/".into(), "Search") }];
    if search_focused {
      keys.push(("^O".into(), if self.view == View::Detail { "Grid view" } else { "Detail view" }));
    } else if self.view == View::Detail {
      keys.extend([("h/l/Arrows".into(), "Browse"), ("Esc".into(), "Grid view")]);
    } else {
      keys.extend([("hjkl/Arrows".into(), "Move"), ("o".into(), "Detail view")]);
    }
    if self.can_act_on_selection() {
      for (key, label) in [("y", "Copy"), ("p", "Copy path"), ("s", "Download")] {
        keys.push((if search_focused { format!("^{}", key.to_uppercase()) } else { key.into() }, label));
      }
    }
    keys.push(("^C".into(), "Quit"));
    keys
  }

  pub fn on_key(&mut self, key: KeyEvent) -> Vec<Effect> {
    let control = key.modifiers.contains(KeyModifiers::CONTROL);
    let search_focused = self.focus == Focus::Search;
    let plain = |c: char| key.code == KeyCode::Char(c) && !control;
    let chord = |c: char| key.code == KeyCode::Char(c) && control;
    // Plain letter bindings only apply outside the search field, where letters are typed instead.
    let bound = |c: char| chord(c) || (!search_focused && plain(c));

    if chord('c') {
      return vec![Effect::Quit];
    }
    if (chord('f') || plain('/')) && !search_focused {
      self.focus = Focus::Search;
      return Vec::new();
    }
    if bound('o') {
      self.toggle_view();
      return self.load_more_if_detail();
    }
    if bound('y') || bound('p') || bound('s') {
      if !self.can_act_on_selection() {
        return Vec::new();
      }
      let filepath = self.selected_result().expect("checked above").filepath.clone();
      return vec![match key.code {
        KeyCode::Char('y') => Effect::CopyImage(filepath),
        KeyCode::Char('p') => Effect::CopyPath(filepath),
        _ => Effect::Download(filepath),
      }];
    }

    if search_focused {
      return self.on_search_key(key);
    }
    match (self.view, key.code) {
      (_, KeyCode::Esc) if self.view == View::Detail => {
        self.toggle_view();
        Vec::new()
      }
      (View::Grid, KeyCode::Enter) => {
        self.toggle_view();
        self.load_more_if_detail()
      }
      (View::Grid, KeyCode::Left | KeyCode::Char('h')) => self.move_by(-1),
      (View::Grid, KeyCode::Right | KeyCode::Char('l')) => self.move_by(1),
      (View::Grid, KeyCode::Down | KeyCode::Char('j')) => self.move_by(self.grid.columns as isize),
      (View::Grid, KeyCode::Char('k')) => self.move_by(-(self.grid.columns as isize)),
      (View::Grid, KeyCode::Up) => {
        if self.selected < self.grid.columns {
          self.focus = Focus::Search;
          Vec::new()
        } else {
          self.move_by(-(self.grid.columns as isize))
        }
      }
      (View::Detail, KeyCode::Left | KeyCode::Char('h')) => self.move_detail(-1),
      (View::Detail, KeyCode::Right | KeyCode::Char('l')) => self.move_detail(1),
      (View::Detail, KeyCode::Up) => {
        self.focus = Focus::Search;
        Vec::new()
      }
      _ => Vec::new(),
    }
  }

  fn on_search_key(&mut self, key: KeyEvent) -> Vec<Effect> {
    match key.code {
      KeyCode::Enter => {
        self.debounce = None;
        self.begin_search(self.input.query().to_string())
      }
      KeyCode::Down | KeyCode::Esc => {
        self.pending_detail_advance = false;
        self.focus = Focus::Results;
        if self.view == View::Grid {
          self.reveal_selection = true;
          self.show_selected_path |= !self.results.is_empty();
        }
        Vec::new()
      }
      _ => {
        if self.input.edit(&key) == Some(true) {
          self.debounce = Some(Instant::now() + DEBOUNCE);
        }
        Vec::new()
      }
    }
  }

  fn load_more_if_detail(&mut self) -> Vec<Effect> {
    if self.view == View::Detail { self.load_more() } else { Vec::new() }
  }

  pub fn toggle_view(&mut self) {
    self.pending_detail_advance = false;
    self.view = if self.view == View::Grid { View::Detail } else { View::Grid };
    if self.view == View::Detail {
      self.show_selected_path = !self.results.is_empty();
    } else {
      self.reveal_selection = true;
    }
  }

  fn move_by(&mut self, offset: isize) -> Vec<Effect> {
    if self.results.is_empty() {
      return Vec::new();
    }
    self.selected = self.selected.saturating_add_signed(offset).min(self.results.len() - 1);
    self.show_selected_path = true;
    self.reveal_selection = true;
    Vec::new()
  }

  pub fn move_detail(&mut self, offset: isize) -> Vec<Effect> {
    self.pending_detail_advance = false;
    if self.results.is_empty() {
      return Vec::new();
    }
    let target = self.selected as isize + offset;
    let more_available = self.loading_more || self.next_cursor.is_some() || !self.remaining.is_empty();
    if target >= self.results.len() as isize && more_available {
      self.pending_detail_advance = true;
      return self.load_more();
    }
    let index = target.clamp(0, self.results.len() as isize - 1) as usize;
    if index == self.selected {
      return Vec::new();
    }
    self.selected = index;
    self.show_selected_path = true;
    self.load_more()
  }

  /// Selects a result clicked in the grid.
  pub fn select(&mut self, index: usize) {
    if index < self.results.len() {
      self.selected = index;
      self.focus = Focus::Results;
      self.show_selected_path = true;
    }
  }

  pub fn scroll_grid(&mut self, lines: isize, grid_lines: usize) {
    let max_scroll = grid_lines.saturating_sub(self.grid.viewport_rows);
    self.scroll = self.scroll.saturating_add_signed(lines).min(max_scroll);
  }

  pub fn copied_path(&mut self, filepath: &str) {
    self.notify(file_name(filepath), "Path copied", false);
  }
}

pub fn file_name(filepath: &str) -> &str {
  filepath.rsplit(['/', std::path::MAIN_SEPARATOR]).next().unwrap_or(filepath)
}

#[cfg(test)]
mod tests {
  use super::*;

  fn key(code: KeyCode) -> KeyEvent {
    KeyEvent::new(code, KeyModifiers::NONE)
  }

  fn ctrl(c: char) -> KeyEvent {
    KeyEvent::new(KeyCode::Char(c), KeyModifiers::CONTROL)
  }

  fn results(count: usize, scored: bool) -> Vec<TuiResult> {
    (0..count)
      .map(|index| TuiResult {
        filepath: format!("/p/{index}.jpg"),
        score: scored.then_some(1.0 - index as f64 / 100.0),
      })
      .collect()
  }

  fn respond(app: &mut App, effects: Vec<Effect>, count: usize, cursor: Option<ImageCursor>) {
    let Some(Effect::Request(generation, request)) = effects.into_iter().next() else { panic!("expected a request") };
    let scored = matches!(request, Request::Search(_));
    app.on_response(Response { generation, request, outcome: Ok((results(count, scored), cursor)) });
  }

  fn type_text(app: &mut App, text: &str) {
    for c in text.chars() {
      app.on_key(key(KeyCode::Char(c)));
    }
  }

  fn cursor() -> Option<ImageCursor> {
    Some(ImageCursor { modified_at: 1.0, filepath: "/p/24.jpg".into() })
  }

  #[test]
  fn starts_by_browsing_with_the_search_focused() {
    let (app, effects) = App::new("~/p".into(), 100);
    assert_eq!(effects, [Effect::Request(1, Request::Browse(None))]);
    assert_eq!(app.focus, Focus::Search);
    assert_eq!(app.status, Some("Searching…"));
  }

  #[test]
  fn typing_debounces_the_search() {
    let (mut app, effects) = App::new("~/p".into(), 100);
    respond(&mut app, effects, 3, None);
    type_text(&mut app, "cat");
    assert_eq!(app.input.text, "cat");
    assert!(app.tick(Instant::now()).is_empty());
    let effects = app.tick(Instant::now() + DEBOUNCE);
    assert_eq!(effects, [Effect::Request(2, Request::Search("cat".into()))]);
  }

  #[test]
  fn letters_are_typed_in_the_search_field_but_bound_elsewhere() {
    let (mut app, effects) = App::new("~/p".into(), 100);
    respond(&mut app, effects, 3, None);
    type_text(&mut app, "jo");
    assert_eq!((app.view, app.input.text.as_str()), (View::Grid, "jo"));
    app.on_key(key(KeyCode::Down));
    assert_eq!(app.focus, Focus::Results);
    app.on_key(key(KeyCode::Char('l')));
    assert_eq!(app.selected, 1);
    app.on_key(key(KeyCode::Char('o')));
    assert_eq!(app.view, View::Detail);
    app.on_key(key(KeyCode::Esc));
    assert_eq!(app.view, View::Grid);
    app.on_key(key(KeyCode::Char('/')));
    assert_eq!(app.focus, Focus::Search);
  }

  #[test]
  fn moves_by_rows_and_returns_to_the_search_from_the_top_row() {
    let (mut app, effects) = App::new("~/p".into(), 100);
    respond(&mut app, effects, 10, None);
    app.grid.columns = 4;
    app.on_key(key(KeyCode::Down));
    app.on_key(key(KeyCode::Char('j')));
    assert_eq!(app.selected, 4);
    app.on_key(key(KeyCode::Up));
    assert_eq!((app.selected, app.focus), (0, Focus::Results));
    app.on_key(key(KeyCode::Up));
    assert_eq!(app.focus, Focus::Search);
  }

  #[test]
  fn ignores_stale_responses() {
    let (mut app, first) = App::new("~/p".into(), 100);
    type_text(&mut app, "dog");
    let second = app.on_key(key(KeyCode::Enter));
    respond(&mut app, first, 5, None);
    assert!(app.results.is_empty());
    respond(&mut app, second, 5, None);
    assert_eq!(app.results.len(), 5);
    assert!(app.ranked);
  }

  #[test]
  fn reveals_search_results_in_batches() {
    let (mut app, _) = App::new("~/p".into(), 2);
    type_text(&mut app, "dog");
    let effects = app.on_key(key(KeyCode::Enter));
    respond(&mut app, effects, 5, None);
    assert_eq!(app.results.len(), 2);
    app.grid.viewport_rows = 10;
    assert!(app.after_frame(0).is_empty());
    assert_eq!(app.results.len(), 4);
    app.after_frame(0);
    app.after_frame(0);
    assert_eq!(app.results.len(), 5);
  }

  #[test]
  fn browsing_pages_with_the_cursor_and_advances_the_detail_view() {
    let (mut app, effects) = App::new("~/p".into(), 100);
    respond(&mut app, effects, 25, cursor());
    app.on_key(key(KeyCode::Down));
    app.on_key(key(KeyCode::Char('o')));
    app.selected = 24;
    let effects = app.on_key(key(KeyCode::Right));
    assert_eq!(effects, [Effect::Request(1, Request::Browse(cursor()))]);
    assert_eq!(app.status, Some("Loading…"));
    respond(&mut app, effects, 25, None);
    assert_eq!((app.results.len(), app.selected), (50, 25));
  }

  #[test]
  fn reports_failed_searches() {
    let (mut app, _) = App::new("~/p".into(), 100);
    type_text(&mut app, "./cat.jpg");
    let effects = app.on_key(key(KeyCode::Enter));
    let Some(Effect::Request(generation, request)) = effects.into_iter().next() else { panic!() };
    app.on_response(Response {
      generation,
      request,
      outcome: Err("interactive mode supports text queries only".into()),
    });
    assert_eq!(app.notifications[0].title, "Search failed");
    assert!(app.notifications[0].error);
    assert_eq!(app.status, None);
  }

  #[test]
  fn shows_no_results() {
    let (mut app, effects) = App::new("~/p".into(), 100);
    respond(&mut app, effects, 0, None);
    assert_eq!(app.status, Some("No results"));
  }

  #[test]
  fn copy_actions_need_a_selected_result_outside_the_search_field() {
    let (mut app, effects) = App::new("~/p".into(), 100);
    respond(&mut app, effects, 2, None);
    assert!(app.on_key(ctrl('y')).is_empty());
    assert!(!app.hotkeys().iter().any(|(_, label)| *label == "Copy"));
    app.on_key(key(KeyCode::Down));
    assert_eq!(app.on_key(key(KeyCode::Char('p'))), [Effect::CopyPath("/p/0.jpg".into())]);
    assert_eq!(app.on_key(ctrl('s')), [Effect::Download("/p/0.jpg".into())]);
    let hotkeys = app.hotkeys();
    assert!(hotkeys.contains(&("y".into(), "Copy")));
    app.on_key(ctrl('f'));
    app.on_key(ctrl('o'));
    assert_eq!(app.on_key(ctrl('y')), [Effect::CopyImage("/p/0.jpg".into())]);
    assert!(app.hotkeys().contains(&("^Y".into(), "Copy")));
  }

  #[test]
  fn quits_on_ctrl_c_anywhere() {
    let (mut app, _) = App::new("~/p".into(), 100);
    assert_eq!(app.on_key(ctrl('c')), [Effect::Quit]);
  }

  #[test]
  fn edits_text_like_a_line_editor() {
    let mut input = Input::default();
    for c in "hello wide world".chars() {
      input.edit(&key(KeyCode::Char(c)));
    }
    input.edit(&ctrl('w'));
    assert_eq!(input.text, "hello wide ");
    input.edit(&key(KeyCode::Home));
    input.edit(&key(KeyCode::Delete));
    assert_eq!(input.text, "ello wide ");
    input.edit(&ctrl('e'));
    input.edit(&key(KeyCode::Backspace));
    assert_eq!(input.text, "ello wide");
    input.edit(&key(KeyCode::Left));
    input.edit(&ctrl('k'));
    assert_eq!(input.text, "ello wid");
    input.edit(&ctrl('u'));
    assert_eq!(input.text, "");
    assert_eq!(input.edit(&key(KeyCode::F(1))), None);
  }
}
