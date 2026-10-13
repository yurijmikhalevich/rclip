//! The interactive terminal UI.

use anyhow::Result;

use crate::db::Db;
use crate::model::Model;
use crate::walk::PathFilter;

pub fn run(_db: Db, _model: Model, _filter: PathFilter, _directory: String, _top_k: usize) -> Result<()> {
  anyhow::bail!("the interactive mode is not implemented yet")
}
