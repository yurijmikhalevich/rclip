//! rclip: AI-powered semantic photo search for the command line.

mod cli;
mod db;
mod imaging;
mod index;
mod model;
mod paths;
mod preview;
mod search;
mod snap;
mod terminal;
mod tui;
mod walk;

use std::io::Write;
use std::process::ExitCode;

use anyhow::{Context, Result};

use cli::{MaxImagePixels, Options};
use db::{Db, VectorCacheReset};
use model::Model;
use model::download::ModelFiles;
use search::{Query, Search, SearchError, SearchResult};
use walk::PathFilter;

fn main() -> ExitCode {
  let options = cli::parse();
  match run(&options) {
    Ok(code) => code,
    Err(error) => {
      eprintln!("rclip: {error:#}");
      ExitCode::FAILURE
    }
  }
}

fn run(options: &Options) -> Result<ExitCode> {
  let args = &options.args;
  let lookup = terminal::env_var;
  let environment = terminal::current_environment(&lookup);
  if args.interactive
    && !args.force_interactive
    && let Err(reason) = terminal::interactive_support(&environment)
  {
    eprintln!("rclip: {reason}");
    return Ok(ExitCode::FAILURE);
  }
  if args.preview
    && !args.force_preview
    && let Err(reason) = terminal::preview_support(&environment)
  {
    eprintln!("rclip: {reason}");
    return Ok(ExitCode::FAILURE);
  }

  let directory = current_directory()?;
  if snap::is_snap() {
    snap::check_permissions(&directory, true);
  }

  let datadir = paths::datadir()?;
  let reset = if args.no_indexing { VectorCacheReset::Forbid } else { VectorCacheReset::Allow };
  let db = Db::open(&datadir.join("db.sqlite3"), reset)?;
  let files = ModelFiles::new(&datadir);
  files.ensure_downloaded()?;
  let model = Model::new(files);
  let filter = PathFilter::new(args.exclude_dir.clone(), args.include_hidden, args.experimental_raw_support);

  if !args.no_indexing {
    let workers = index::loading_workers();
    let max_pixels = match args.max_image_megapixels {
      MaxImagePixels::Auto => Some(imaging::default_max_image_pixels(workers)),
      MaxImagePixels::Disabled => None,
      MaxImagePixels::Limit(limit) => Some(limit),
    };
    let index_options = index::IndexOptions {
      batch_size: args.indexing_batch_size,
      raw_support: args.experimental_raw_support,
      max_pixels,
      workers,
    };
    index::ensure_index(&db, || model.index_encoder(), &filter, &directory, &index_options)?;
  }

  if args.interactive {
    tui::run(db, model, filter, directory, options.top_k)?;
    return Ok(ExitCode::SUCCESS);
  }

  let search = Search { db: &db, model: &model, filter: &filter, directory: &directory };
  let query = Query { query: args.query.as_deref().unwrap_or_default(), add: &args.add, subtract: &args.subtract };
  match search.search(&query, options.top_k, None) {
    Ok(results) => print_results(&results, options),
    Err(SearchError::Query(error)) => {
      eprintln!("{error}");
      Ok(ExitCode::FAILURE)
    }
    Err(error) => Err(error.into()),
  }
}

/// The working directory as a string, which is also how paths are stored in the index.
fn current_directory() -> Result<String> {
  let directory = std::env::current_dir().context("cannot get the current directory")?;
  directory.into_os_string().into_string().map_err(|path| anyhow::anyhow!("{} is not valid UTF-8", path.display()))
}

/// Prints the results, failing when a preview cannot be shown after showing the remaining ones.
fn print_results(results: &[SearchResult], options: &Options) -> Result<ExitCode> {
  let stdout = std::io::stdout();
  let mut output = std::io::BufWriter::new(stdout.lock());
  let mut code = ExitCode::SUCCESS;
  if options.args.null {
    for result in results {
      write!(output, "{}\0", result.filepath)?;
    }
  } else if options.args.json {
    let results: Vec<JsonResult<'_>> = results
      .iter()
      .map(|result| JsonResult { score: (result.score * 1000.0).round() / 1000.0, filepath: &result.filepath })
      .collect();
    serde_json::to_writer(&mut output, &results)?;
    writeln!(output)?;
  } else if options.args.filepath_only {
    for result in results {
      writeln!(output, "{}", result.filepath)?;
    }
  } else {
    writeln!(output, "score\tfilepath")?;
    for result in results {
      writeln!(output, "{:.3}\t\"{}\"", result.score, result.filepath)?;
      if options.args.preview {
        output.flush()?;
        if let Err(error) = preview::preview(&mut output, &result.filepath, options.args.preview_height) {
          output.flush()?;
          eprintln!("rclip: cannot preview {}: {error:#}", result.filepath);
          code = ExitCode::FAILURE;
        }
      }
    }
  }
  output.flush()?;
  Ok(code)
}

#[derive(serde::Serialize)]
struct JsonResult<'a> {
  score: f64,
  filepath: &'a str,
}
