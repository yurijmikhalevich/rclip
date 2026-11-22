//! The interactive terminal UI: a searchable image grid and a detail view, drawn with ratatui and
//! the Kitty graphics protocol.

mod app;
mod images;
mod transfer;
mod ui;
mod worker;

use std::io::{Stdout, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::Result;
use crossbeam_channel::{Receiver, Sender, select, unbounded};
use crossterm::event::{
  self, DisableMouseCapture, EnableMouseCapture, Event as TerminalEvent, KeyEventKind, MouseButton, MouseEvent,
  MouseEventKind,
};
use crossterm::execute;
use crossterm::terminal::{self, EnterAlternateScreen, LeaveAlternateScreen};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Position, Rect};

use crate::db::Db;
use crate::model::Model;
use crate::walk::PathFilter;
use app::{App, Effect, View, file_name};
use images::{ImageKey, Images, Prepared};
use worker::{Response, Worker};

const DOUBLE_CLICK: Duration = Duration::from_millis(500);
const SCROLL_LINES: isize = 3;

enum Event {
  Terminal(TerminalEvent),
  Search(Response),
  Image(ImageKey, Result<Prepared, String>),
}

pub fn run(db: Db, model: Model, filter: PathFilter, directory: String, top_k: usize) -> Result<()> {
  let (events, events_rx) = unbounded();
  let mut worker =
    Worker::spawn(worker::Context { db, model, filter, directory: directory.clone() }, events.clone(), Event::Search);
  let mut images = Images::new(events.clone(), Event::Image);
  let input = InputReader::spawn(events);

  let mut screen = Screen::enter()?;
  let result =
    event_loop(&mut screen, &input, &events_rx, &mut worker, &mut images, display_directory(&directory), top_k);
  images.clear(&mut std::io::stdout()).ok();
  screen.leave();
  input.stop();
  result
}

fn display_directory(directory: &str) -> String {
  match crate::paths::home_dir()
    .and_then(|home| std::path::Path::new(directory).strip_prefix(home).ok().map(|p| p.to_owned()))
  {
    Some(relative) if relative.as_os_str().is_empty() => "~".into(),
    Some(relative) => format!("~{}{}", std::path::MAIN_SEPARATOR, relative.display()),
    None => directory.to_string(),
  }
}

struct Session<'a> {
  screen: &'a mut Screen,
  input: &'a InputReader,
  worker: &'a mut Worker,
  images: &'a mut Images,
  app: App,
  hits: ui::HitMap,
  last_click: Option<(Instant, Click)>,
  cell_pixels: (u16, u16),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Click {
  Card(usize),
  Detail,
}

fn event_loop(
  screen: &mut Screen,
  input: &InputReader,
  events: &Receiver<Event>,
  worker: &mut Worker,
  images: &mut Images,
  directory: String,
  top_k: usize,
) -> Result<()> {
  let (app, effects) = App::new(directory, top_k);
  let mut session = Session {
    screen,
    input,
    worker,
    images,
    app,
    hits: ui::HitMap::default(),
    last_click: None,
    cell_pixels: cell_pixels(),
  };
  if session.apply(effects)? {
    return Ok(());
  }
  loop {
    session.draw()?;
    let effects = session.app.after_frame(session.hits.grid_lines);
    if !effects.is_empty() {
      if session.apply(effects)? {
        return Ok(());
      }
      continue;
    }
    // Wait for the next event or timer, then handle everything that is already queued so that a
    // burst of loaded images causes a single redraw.
    let timeout = session.app.next_deadline().map(|deadline| deadline.saturating_duration_since(Instant::now()));
    let first = match timeout {
      Some(timeout) => select! { recv(events) -> event => event.ok(), default(timeout) => None },
      None => events.recv().ok(),
    };
    let mut effects = session.app.tick(Instant::now());
    for event in first.into_iter().chain(std::iter::from_fn(|| events.try_recv().ok())) {
      effects.extend(session.handle(event));
    }
    if session.apply(effects)? {
      return Ok(());
    }
  }
}

impl Session<'_> {
  fn draw(&mut self) -> Result<()> {
    self.images.flush(&mut std::io::stdout())?;
    self.images.begin_frame();
    let Session { screen, app, images, hits, cell_pixels, .. } = self;
    screen.terminal.draw(|frame| {
      *hits = ui::draw(frame, app, &mut ui::Context { images, cell_pixels: *cell_pixels });
    })?;
    self.images.end_frame();
    self.images.flush(&mut std::io::stdout())?;
    Ok(())
  }

  fn handle(&mut self, event: Event) -> Vec<Effect> {
    match event {
      Event::Search(response) => self.app.on_response(response),
      Event::Image(key, result) => {
        self.images.loaded(key, result);
        Vec::new()
      }
      Event::Terminal(TerminalEvent::Key(key)) if key.kind != KeyEventKind::Release => self.app.on_key(key),
      Event::Terminal(TerminalEvent::Mouse(mouse)) => self.on_mouse(mouse),
      Event::Terminal(TerminalEvent::Resize(..)) => {
        self.cell_pixels = cell_pixels();
        Vec::new()
      }
      Event::Terminal(_) => Vec::new(),
    }
  }

  fn on_mouse(&mut self, mouse: MouseEvent) -> Vec<Effect> {
    let position = Position::new(mouse.column, mouse.row);
    let inside = |area: Rect| area.contains(position);
    match mouse.kind {
      MouseEventKind::ScrollDown | MouseEventKind::ScrollUp if inside(self.hits.grid) => {
        let lines = if mouse.kind == MouseEventKind::ScrollDown { SCROLL_LINES } else { -SCROLL_LINES };
        self.app.scroll_grid(lines, self.hits.grid_lines);
        Vec::new()
      }
      MouseEventKind::Down(MouseButton::Left) => {
        if inside(self.hits.input) {
          self.app.focus = app::Focus::Search;
          return Vec::new();
        }
        if let Some(&(_, offset)) = self.hits.filmstrip.iter().find(|(area, _)| inside(*area)) {
          return self.app.move_detail(offset);
        }
        let click = if let Some(&(_, index)) = self.hits.cards.iter().find(|(area, _)| inside(*area)) {
          self.app.select(index);
          Click::Card(index)
        } else if self.app.view == View::Detail && inside(self.hits.detail_image) {
          Click::Detail
        } else {
          return Vec::new();
        };
        let now = Instant::now();
        let double = self.last_click.is_some_and(|(at, last)| last == click && now - at < DOUBLE_CLICK);
        self.last_click = if double { None } else { Some((now, click)) };
        if double {
          self.app.focus = app::Focus::Results;
          self.app.toggle_view();
        }
        Vec::new()
      }
      _ => Vec::new(),
    }
  }

  /// Carries out effects; returns `true` when the app should quit.
  fn apply(&mut self, effects: Vec<Effect>) -> Result<bool> {
    for effect in effects {
      match effect {
        Effect::Quit => return Ok(true),
        Effect::Request(generation, request) => self.worker.request(generation, request),
        Effect::CopyPath(filepath) => {
          let mut stdout = std::io::stdout();
          stdout.write_all(transfer::osc52_copy(&filepath).as_bytes())?;
          stdout.flush()?;
          self.app.copied_path(&filepath);
        }
        Effect::CopyImage(filepath) => {
          let result = transfer::kitten_executable().and_then(|_| {
            self.suspended("pausing the TUI to copy the image…", || transfer::copy_image_to_clipboard(&filepath))
          });
          match result {
            Ok(()) => self.app.notify(file_name(&filepath), "Image copied", false),
            Err(error) => self.app.notify("Unable to copy image", error, true),
          }
        }
        Effect::Download(filepath) => {
          if !transfer::is_remote_session() {
            self.app.notify("Image is already local", filepath, false);
            continue;
          }
          let result = transfer::kitten_executable().and_then(|_| {
            self.suspended("pausing the TUI to download the image…", || transfer::download_image(&filepath))
          });
          match result {
            Ok(()) => self.app.notify(file_name(&filepath), "Saved to ~/Downloads", false),
            Err(error) => self.app.notify("Unable to download image", error, true),
          }
        }
      }
    }
    Ok(false)
  }

  /// Hands the terminal to `run` (which talks to Kitty through `kitten`), explaining the visible
  /// screen switch, and restores the UI afterwards.
  fn suspended<T>(&mut self, message: &str, run: impl FnOnce() -> T) -> T {
    self.input.pause();
    self.images.clear(&mut std::io::stdout()).ok();
    self.screen.leave();
    println!("rclip: {message}");
    let result = run();
    // `kitten query_terminal` ends its DCS query with BEL, which leaves iTerm2 parsing it; terminate
    // it before switching back to the alternate screen.
    print!("\x1b\\");
    std::io::stdout().flush().ok();
    if let Err(error) = self.screen.reenter() {
      self.app.notify("Unable to restore the terminal", error.to_string(), true);
    }
    self.input.resume();
    result
  }
}

/// The terminal's cell size in pixels, falling back to a common size when it is not reported.
fn cell_pixels() -> (u16, u16) {
  match terminal::window_size() {
    Ok(size) if size.width > 0 && size.height > 0 && size.columns > 0 && size.rows > 0 => {
      ((size.width / size.columns).max(1), (size.height / size.rows).max(1))
    }
    _ => (10, 20),
  }
}

/// The terminal in raw mode on the alternate screen.
struct Screen {
  terminal: Terminal<CrosstermBackend<Stdout>>,
  active: bool,
}

impl Screen {
  fn enter() -> Result<Self> {
    let previous_hook = std::panic::take_hook();
    // Image loaders turn their panics into errors shown in place of the image, so only a panic on
    // the main thread ends the interface.
    std::panic::set_hook(Box::new(move |info| {
      if std::thread::current().name() == Some("main") {
        restore_terminal();
        previous_hook(info);
      }
    }));
    // An active screen restores the terminal when dropped, which also undoes a setup that failed halfway.
    let screen = Screen { terminal: Terminal::new(CrosstermBackend::new(std::io::stdout()))?, active: true };
    setup_terminal()?;
    Ok(screen)
  }

  fn leave(&mut self) {
    if self.active {
      restore_terminal();
      self.active = false;
    }
  }

  fn reenter(&mut self) -> Result<()> {
    self.active = true;
    setup_terminal()?;
    // Fresh buffers match the freshly cleared screen, so the next frame redraws everything.
    self.terminal = Terminal::new(CrosstermBackend::new(std::io::stdout()))?;
    Ok(())
  }
}

impl Drop for Screen {
  fn drop(&mut self) {
    self.leave();
  }
}

fn setup_terminal() -> Result<()> {
  terminal::enable_raw_mode()?;
  // Clearing directly avoids ratatui's clear, which queries the cursor position and needs a reply.
  execute!(std::io::stdout(), EnterAlternateScreen, EnableMouseCapture, terminal::Clear(terminal::ClearType::All))?;
  Ok(())
}

fn restore_terminal() {
  execute!(std::io::stdout(), DisableMouseCapture, LeaveAlternateScreen, crossterm::cursor::Show).ok();
  terminal::disable_raw_mode().ok();
}

/// Reads terminal input on its own thread. It can be paused so that another program (`kitten`)
/// can talk to the terminal without its responses being stolen.
struct InputReader {
  paused: Arc<AtomicBool>,
  /// Set by the reader once it has stopped reading after a pause request.
  idle: Arc<AtomicBool>,
  stop: Arc<AtomicBool>,
}

impl InputReader {
  fn spawn(events: Sender<Event>) -> Self {
    let (paused, idle, stop) =
      (Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(false)));
    let reader = InputReader { paused: Arc::clone(&paused), idle: Arc::clone(&idle), stop: Arc::clone(&stop) };
    std::thread::spawn(move || {
      read_input(&events, &paused, &idle, &stop);
      // A reader that has stopped never blocks a pause.
      idle.store(true, Ordering::Release);
    });
    reader
  }

  fn pause(&self) {
    self.paused.store(true, Ordering::Release);
    while !self.idle.load(Ordering::Acquire) {
      std::thread::sleep(Duration::from_millis(5));
    }
  }

  fn resume(&self) {
    self.paused.store(false, Ordering::Release);
  }

  fn stop(&self) {
    self.stop.store(true, Ordering::Relaxed);
  }
}

fn read_input(events: &Sender<Event>, paused: &AtomicBool, idle: &AtomicBool, stop: &AtomicBool) {
  while !stop.load(Ordering::Relaxed) {
    if paused.load(Ordering::Acquire) {
      idle.store(true, Ordering::Release);
      std::thread::sleep(Duration::from_millis(10));
      continue;
    }
    idle.store(false, Ordering::Release);
    match event::poll(Duration::from_millis(50)) {
      Ok(true) => match event::read() {
        Ok(event) => {
          if events.send(Event::Terminal(event)).is_err() {
            return;
          }
        }
        Err(_) => return,
      },
      Ok(false) => {}
      Err(_) => return,
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn abbreviates_the_home_directory() {
    let home = crate::paths::home_dir().expect("the tests run with a home directory");
    let home = home.to_str().expect("the home directory is valid UTF-8");
    let separator = std::path::MAIN_SEPARATOR;
    assert_eq!(display_directory(home), "~");
    assert_eq!(display_directory(&format!("{home}{separator}photos")), format!("~{separator}photos"));
    // A directory next to the home directory only shares its beginning.
    let elsewhere = format!("{home}-elsewhere{separator}photos");
    assert_eq!(display_directory(&elsewhere), elsewhere);
  }
}
