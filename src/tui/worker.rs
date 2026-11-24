//! Runs searches and browse queries off the UI thread.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crossbeam_channel::{Receiver, Sender, unbounded};

use crate::db::{Db, ImageCursor};
use crate::model::{Model, is_text_query};
use crate::search::{Query, Search, SearchError};
use crate::walk::PathFilter;

/// Images listed per page when browsing.
pub const BROWSE_PAGE_SIZE: usize = 25;
/// More than anyone scrolls through, while keeping search as fast and memory-flat as the CLI's on large libraries.
const MAX_SEARCH_RESULTS: usize = 1000;

#[derive(Clone, Debug, PartialEq)]
pub enum Request {
  /// Rank the images against a text query.
  Search(String),
  /// List images, most recently modified first, after the cursor.
  Browse(Option<ImageCursor>),
}

#[derive(Debug, PartialEq)]
pub struct TuiResult {
  pub filepath: String,
  pub score: Option<f64>,
}

#[derive(Debug)]
pub struct Response {
  pub generation: u64,
  pub request: Request,
  pub outcome: Result<(Vec<TuiResult>, Option<ImageCursor>), String>,
}

pub struct Worker {
  requests: Option<Sender<(u64, Request, Arc<AtomicBool>)>>,
  cancel: Arc<AtomicBool>,
  thread: Option<std::thread::JoinHandle<()>>,
}

pub struct Context {
  pub db: Db,
  pub model: Model,
  pub filter: PathFilter,
  pub directory: String,
}

impl Worker {
  pub fn spawn<E: Send + 'static>(context: Context, events: Sender<E>, wrap: fn(Response) -> E) -> Self {
    let (requests, requests_rx) = unbounded();
    let thread =
      std::thread::spawn(move || run(&context, &requests_rx, |response| events.send(wrap(response)).is_ok()));
    Worker { requests: Some(requests), cancel: Arc::new(AtomicBool::new(false)), thread: Some(thread) }
  }

  /// Starts a request, cancelling the one in progress.
  pub fn request(&mut self, generation: u64, request: Request) {
    self.cancel.store(true, Ordering::Relaxed);
    self.cancel = Arc::new(AtomicBool::new(false));
    if let Some(requests) = &self.requests {
      requests.send((generation, request, Arc::clone(&self.cancel))).ok();
    }
  }
}

impl Drop for Worker {
  fn drop(&mut self) {
    self.cancel.store(true, Ordering::Relaxed);
    // Closing the channel stops the thread once the current request finishes.
    self.requests = None;
    if let Some(thread) = self.thread.take() {
      thread.join().ok();
    }
  }
}

fn run(context: &Context, requests: &Receiver<(u64, Request, Arc<AtomicBool>)>, send: impl Fn(Response) -> bool) {
  while let Ok(mut next) = requests.recv() {
    // Only the newest request matters; older ones have been cancelled.
    while let Ok(newer) = requests.try_recv() {
      next = newer;
    }
    let (generation, request, cancel) = next;
    if cancel.load(Ordering::Relaxed) {
      continue;
    }
    let outcome = handle(context, &request, &cancel);
    if matches!(outcome, Err(SearchError::Cancelled(_))) || cancel.load(Ordering::Relaxed) {
      continue;
    }
    let outcome = outcome.map_err(|error| error.to_string());
    if !send(Response { generation, request, outcome }) {
      return;
    }
  }
}

fn handle(
  context: &Context,
  request: &Request,
  cancel: &AtomicBool,
) -> Result<(Vec<TuiResult>, Option<ImageCursor>), SearchError> {
  let search =
    Search { db: &context.db, model: &context.model, filter: &context.filter, directory: &context.directory };
  match request {
    Request::Search(query) => {
      if !is_text_query(query) {
        return Err(SearchError::Other(anyhow::anyhow!("interactive mode supports text queries only")));
      }
      let results = search.search(&Query { query, add: &[], subtract: &[] }, MAX_SEARCH_RESULTS, Some(cancel))?;
      Ok((
        results.into_iter().map(|result| TuiResult { filepath: result.filepath, score: Some(result.score) }).collect(),
        None,
      ))
    }
    Request::Browse(after) => {
      let (paths, cursor) = search.list_images(BROWSE_PAGE_SIZE, after.as_ref(), Some(cancel))?;
      Ok((paths.into_iter().map(|filepath| TuiResult { filepath, score: None }).collect(), cursor))
    }
  }
}
