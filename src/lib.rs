//! rclip: AI-powered semantic photo search for the command line.
//!
//! The library holds everything the `rclip` binary and the benchmarks in `examples/` share.

pub mod cli;
pub mod db;
pub mod imaging;
pub mod index;
mod kitty;
pub mod model;
pub mod paths;
pub mod preview;
pub mod search;
pub mod snap;
pub mod terminal;
pub mod tui;
pub mod walk;
