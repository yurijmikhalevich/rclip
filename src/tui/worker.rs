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

#[cfg(test)]
mod tests {
  use super::*;
  use crate::db::NewImage;
  use crate::model::download::ModelFiles;

  fn native(path: &str) -> String {
    path.replace('/', std::path::MAIN_SEPARATOR_STR)
  }

  /// A library of `count` images, the first one modified last, with a model that cannot be loaded.
  fn context(count: usize) -> Context {
    let db = Db::open_in_memory().unwrap();
    for index in 0..count {
      let filepath = native(&format!("/p/{index:02}.jpg"));
      let modified_at = (count - index) as f64;
      db.upsert_image(&NewImage { filepath: &filepath, modified_at, size: 1, vector: &[0; 4], hash: "h" }).unwrap();
    }
    Context {
      db,
      model: Model::new(ModelFiles::new(std::path::Path::new("/nonexistent"))),
      filter: PathFilter::new(None, false, false),
      directory: native("/p"),
    }
  }

  /// Queues `requests` (generation, request, whether it was cancelled) and returns the worker's answers.
  fn answers(context: &Context, requests: Vec<(u64, Request, bool)>) -> Vec<Response> {
    let (sender, receiver) = unbounded();
    for (generation, request, cancelled) in requests {
      sender.send((generation, request, Arc::new(AtomicBool::new(cancelled)))).unwrap();
    }
    drop(sender);
    let responses = std::cell::RefCell::new(Vec::new());
    run(context, &receiver, |response| {
      responses.borrow_mut().push(response);
      true
    });
    responses.into_inner()
  }

  #[test]
  fn browses_a_page_at_a_time() {
    let context = context(BROWSE_PAGE_SIZE + 1);
    let cancel = AtomicBool::new(false);
    let (page, cursor) = handle(&context, &Request::Browse(None), &cancel).unwrap();
    assert_eq!(page.len(), BROWSE_PAGE_SIZE);
    assert_eq!(page[0], TuiResult { filepath: native("/p/00.jpg"), score: None });
    assert!(cursor.is_some());
    let (page, cursor) = handle(&context, &Request::Browse(cursor), &cancel).unwrap();
    assert_eq!(page, [TuiResult { filepath: native("/p/25.jpg"), score: None }]);
    assert_eq!(cursor, None);
  }

  #[test]
  fn rejects_image_queries() {
    for query in ["./cat.jpg", "2:./cat.jpg", "https://example.com/cat.jpg"] {
      let responses = answers(&context(1), vec![(1, Request::Search(query.into()), false)]);
      assert_eq!(responses[0].outcome, Err("interactive mode supports text queries only".into()), "{query}");
      assert_eq!((responses[0].generation, &responses[0].request), (1, &Request::Search(query.into())));
    }
  }

  #[test]
  fn answers_only_the_newest_request() {
    let requests = vec![
      (1, Request::Browse(None), false),
      (2, Request::Search("./cat.jpg".into()), false),
      (3, Request::Browse(None), false),
    ];
    let responses = answers(&context(2), requests);
    assert_eq!(responses.len(), 1);
    assert_eq!(responses[0].generation, 3);
    assert_eq!(responses[0].outcome.as_ref().map(|(results, _)| results.len()), Ok(2));
  }

  #[test]
  fn does_not_answer_cancelled_requests() {
    let context = context(2);
    assert!(answers(&context, vec![(1, Request::Browse(None), true)]).is_empty());
    // A request cancelled while it runs stops with an error instead of finishing.
    let interrupted = handle(&context, &Request::Browse(None), &AtomicBool::new(true));
    assert!(matches!(interrupted, Err(SearchError::Cancelled(_))));
  }

  #[test]
  fn a_new_request_cancels_the_previous_one() {
    let (events, received) = unbounded();
    let mut worker = Worker::spawn(context(2), events, std::convert::identity);
    worker.request(1, Request::Browse(None));
    let first = Arc::clone(&worker.cancel);
    worker.request(2, Request::Browse(None));
    assert!(first.load(Ordering::Relaxed));
    assert!(!worker.cancel.load(Ordering::Relaxed));
    // The first request may have finished before it was cancelled; the newest one is always answered.
    let newest = std::iter::from_fn(|| received.recv_timeout(std::time::Duration::from_secs(30)).ok())
      .find(|response: &Response| response.generation == 2)
      .expect("the newest request is answered");
    assert_eq!(newest.outcome.map(|(results, _)| results.len()), Ok(2));
  }
}
