#!/usr/bin/env python3
"""
Merge two rclip SQLite databases with progress tracking and performance optimizations.

Usage: python merge_rclip_dbs.py <db1.db> <db2.db> <output.db>
"""

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Dict, Tuple
from tqdm import tqdm

__version__ = "1.0.0"


class RclipDBMerger:
  """Merge two rclip databases with performance optimizations."""

  def __init__(
    self,
    db1_path: str,
    db2_path: str,
    output_path: str,
    verbose: bool = False,
    batch_size: int = 10000,
    commit_interval: int = 50000,
    cache_size_mb: int = 64,
    mmap_size_mb: int = 1024,
    dry_run: bool = False,
  ):
    self.db1_path = Path(db1_path)
    self.db2_path = Path(db2_path)
    self.output_path = Path(output_path)
    self.verbose = verbose
    self.batch_size = batch_size
    self.commit_interval = commit_interval
    self.cache_size_mb = cache_size_mb
    self.mmap_size_mb = mmap_size_mb
    self.dry_run = dry_run

    # Statistics tracking
    self.stats = {
      "db1_images": 0,
      "db2_images": 0,
      "merged_images": 0,
      "conflicts_resolved": 0,
      "deleted_preserved": 0,
      "start_time": time.time(),
    }

  def log(self, message: str):
    """Print message if verbose mode is enabled."""
    if self.verbose:
      print(f"[{time.strftime('%H:%M:%S')}] {message}")

  def validate_databases(self) -> Tuple[int, int]:
    """Validate input databases and get versions."""
    for db_path in [self.db1_path, self.db2_path]:
      if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

      # Check database integrity
      if not self._check_database_integrity(db_path):
        raise ValueError(f"Database integrity check failed: {db_path}")

    # Check versions
    version1 = self._get_db_version(self.db1_path)
    version2 = self._get_db_version(self.db2_path)

    if version1 != version2:
      raise ValueError(f"Database version mismatch: {version1} vs {version2}")

    return version1, version2

  def _check_database_integrity(self, db_path: Path) -> bool:
    """Check database integrity using SQLite PRAGMA."""
    con = sqlite3.connect(db_path)
    try:
      result = con.execute("PRAGMA integrity_check").fetchone()
      return result[0] == "ok"
    except Exception:
      return False
    finally:
      con.close()

  def _get_db_version(self, db_path: Path) -> int:
    """Get database version."""
    con = sqlite3.connect(db_path)
    try:
      result = con.execute("SELECT version FROM db_version").fetchone()
      return result[0] if result else 1
    finally:
      con.close()

  def _count_images(self, db_path: Path) -> int:
    """Count total images in database."""
    con = sqlite3.connect(db_path)
    try:
      result = con.execute("SELECT COUNT(*) FROM images").fetchone()
      return result[0] if result else 0
    finally:
      con.close()

  def _create_output_database(self, version: int):
    """Create output database with schema."""
    self.log("Creating output database...")

    # Ensure parent directory exists
    self.output_path.parent.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(self.output_path)

    # Drop existing tables if they exist (for overwrite)
    con.execute("DROP TABLE IF EXISTS images")
    con.execute("DROP TABLE IF EXISTS db_version")

    # Create schema without indexes first (faster bulk insert)
    con.execute("""
            CREATE TABLE images (
                id INTEGER PRIMARY KEY,
                deleted BOOLEAN,
                filepath TEXT NOT NULL,
                modified_at DATETIME NOT NULL,
                size INTEGER NOT NULL,
                vector BLOB NOT NULL,
                indexing BOOLEAN
            )
        """)

    con.execute("CREATE TABLE db_version (version INTEGER)")
    con.execute("INSERT INTO db_version(version) VALUES (?)", (version,))

    con.commit()
    con.close()

  def _optimize_connection(self, con: sqlite3.Connection):
    """Apply performance optimizations to connection."""
    # Performance pragmas
    cache_kb = -1 * self.cache_size_mb * 1024  # Negative value = size in KB
    con.execute(f"PRAGMA cache_size = {cache_kb}")
    con.execute("PRAGMA synchronous = OFF")  # Faster, less safe
    con.execute("PRAGMA journal_mode = WAL")  # Write-ahead logging
    con.execute("PRAGMA temp_store = MEMORY")  # Use memory for temp tables
    mmap_bytes = self.mmap_size_mb * 1024 * 1024
    con.execute(f"PRAGMA mmap_size = {mmap_bytes}")  # Memory-mapped I/O size

  def merge(self):
    """Perform the merge operation with progress tracking."""
    print(f"\nrclip-db-merge v{__version__}")
    print("\nMerging rclip databases:")
    print(f"  Database 1: {self.db1_path}")
    print(f"  Database 2: {self.db2_path}")
    print(f"  Output: {self.output_path}")
    print()

    # Validate databases
    self.log("Validating databases...")
    version1, version2 = self.validate_databases()

    # Count images for progress tracking
    self.stats["db1_images"] = self._count_images(self.db1_path)
    self.stats["db2_images"] = self._count_images(self.db2_path)
    total_images = self.stats["db1_images"] + self.stats["db2_images"]

    print(f"Database version: {version1}")
    print(f"Database 1: {self.stats['db1_images']:,} images")
    print(f"Database 2: {self.stats['db2_images']:,} images")
    print(f"Total to process: {total_images:,} images")

    if self.dry_run:
      print("\nDRY RUN MODE - No output database will be created")
      print(f"\nWould merge {total_images:,} total images")
      print(f"Output would be: {self.output_path}")
      return

    print()

    # Create output database
    self._create_output_database(version1)

    # Open connections
    out_con = sqlite3.connect(self.output_path)
    out_con.row_factory = sqlite3.Row
    self._optimize_connection(out_con)

    # Create filepath index for conflict detection
    self.log("Building filepath index...")
    filepath_index = {}

    # Process databases
    progress = tqdm(total=total_images, desc="Merging databases", unit="rows")

    try:
      # First pass: Build complete index with conflict resolution
      self._build_filepath_index(self.db1_path, filepath_index, progress, "DB1")
      self._build_filepath_index(self.db2_path, filepath_index, progress, "DB2")

      # Second pass: Insert winning records
      self._insert_from_index(out_con, filepath_index)

      progress.close()

      # Create indexes after bulk insert
      print("\nCreating indexes...")
      self._create_indexes(out_con)

      # Final commit and optimize
      print("Optimizing database...")
      out_con.commit()
      out_con.execute("VACUUM")
      out_con.execute("ANALYZE")

    finally:
      out_con.close()

    # Print summary
    self._print_summary()

    # Validate output
    if not self._validate_output():
      print("\nWarning: Output validation failed - please verify the merged database")

  def _build_filepath_index(self, db_path: Path, filepath_index: Dict[str, dict], progress: tqdm, db_label: str):
    """Build index from database, resolving conflicts."""
    in_con = sqlite3.connect(db_path)
    in_con.row_factory = sqlite3.Row

    cursor = in_con.execute("""
        SELECT id, deleted, filepath, modified_at, size, vector, indexing 
        FROM images 
        ORDER BY id
    """)

    for row in cursor:
      filepath = row["filepath"]
      norm_filepath = os.path.normpath(filepath)

      # Check for conflicts
      if norm_filepath in filepath_index:
        # Resolve conflict: keep newer or deleted
        existing = filepath_index[norm_filepath]
        if row["deleted"] or (not existing["deleted"] and row["modified_at"] > existing["modified_at"]):
          # This row wins
          filepath_index[norm_filepath] = dict(row)
          self.stats["conflicts_resolved"] += 1
          if row["deleted"]:
            self.stats["deleted_preserved"] += 1
      else:
        # New filepath
        filepath_index[norm_filepath] = dict(row)

      progress.update(1)

    in_con.close()
    self.log(f"Indexed {db_label}")

  def _insert_from_index(self, out_con: sqlite3.Connection, filepath_index: Dict[str, dict]):
    """Insert all records from the index."""
    batch = []

    for filepath, row in filepath_index.items():
      batch.append(row)

      if len(batch) >= self.batch_size:
        self._insert_batch(out_con, batch)
        batch = []
        out_con.commit()

    # Insert remaining batch
    if batch:
      self._insert_batch(out_con, batch)
      out_con.commit()

    self.stats["merged_images"] = len(filepath_index)

  def _insert_batch(self, con: sqlite3.Connection, batch: list):
    """Insert a batch of records efficiently."""
    if not batch:
      return

    con.executemany(
      """
            INSERT INTO images (deleted, filepath, modified_at, size, vector, indexing)
            VALUES (:deleted, :filepath, :modified_at, :size, :vector, :indexing)
        """,
      batch,
    )

    self.stats["merged_images"] += len(batch)

  def _create_indexes(self, con: sqlite3.Connection):
    """Create indexes after bulk insert."""
    con.execute("CREATE UNIQUE INDEX IF NOT EXISTS existing_images ON images(filepath) WHERE deleted IS NULL")
    con.commit()

  def check_versions_only(self):
    """Check and print database versions without merging."""
    print(f"\nrclip-db-merge v{__version__} - Database Version Check")
    print("=" * 50)

    for db_path, label in [(self.db1_path, "Database 1"), (self.db2_path, "Database 2")]:
      if not db_path.exists():
        print(f"\n{label}: {db_path}")
        print("  Status: NOT FOUND")
        continue

      version = self._get_db_version(db_path)
      image_count = self._count_images(db_path)
      size_mb = db_path.stat().st_size / 1024 / 1024

      print(f"\n{label}: {db_path}")
      print(f"  Version: {version}")
      print(f"  Images: {image_count:,}")
      print(f"  Size: {size_mb:.1f} MB")

    # Check compatibility
    if self.db1_path.exists() and self.db2_path.exists():
      try:
        version1, version2 = self.validate_databases()
        print(f"\n✓ Databases are compatible for merging (both version {version1})")
      except ValueError as e:
        print(f"\n✗ {e}")

  def _validate_output(self) -> bool:
    """Basic validation of merged database."""
    try:
      con = sqlite3.connect(self.output_path)
      # Check we have images
      count = con.execute("SELECT COUNT(*) FROM images").fetchone()[0]
      # Check no duplicate filepaths in non-deleted images
      duplicates = con.execute("""
        SELECT COUNT(*) FROM (
          SELECT filepath FROM images 
          WHERE deleted IS NULL 
          GROUP BY filepath 
          HAVING COUNT(*) > 1
        )
      """).fetchone()[0]
      con.close()
      return count > 0 and duplicates == 0
    except Exception:
      return False

  def _print_summary(self):
    """Print merge summary."""
    elapsed = time.time() - self.stats["start_time"]

    print("\n" + "=" * 60)
    print("MERGE COMPLETE")
    print("=" * 60)
    print(f"Total images merged: {self.stats['merged_images']:,}")
    print(f"Conflicts resolved: {self.stats['conflicts_resolved']:,}")
    print(f"Deleted images preserved: {self.stats['deleted_preserved']:,}")
    print(f"Time elapsed: {elapsed:.1f} seconds")
    print(f"Processing rate: {self.stats['merged_images'] / elapsed:.0f} images/second")
    print(f"Output database: {self.output_path}")
    print(f"Output size: {self.output_path.stat().st_size / 1024 / 1024:.1f} MB")


def main():
  parser = argparse.ArgumentParser(
    description=f"Merge two rclip SQLite databases (v{__version__})",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  %(prog)s db1.db db2.db merged.db
  %(prog)s ~/rclip1.db ~/rclip2.db ~/rclip_merged.db -v
        """,
  )

  parser.add_argument("db1", help="First database file")
  parser.add_argument("db2", help="Second database file")
  parser.add_argument("output", help="Output merged database file")
  parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
  parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
  parser.add_argument("--force", action="store_true", help="Overwrite output file if it exists")
  parser.add_argument(
    "--batch-size", type=int, default=10000, help="Number of rows to process in each batch (default: 10000)"
  )
  parser.add_argument(
    "--commit-interval", type=int, default=50000, help="Number of rows between commits (default: 50000)"
  )
  parser.add_argument("--cache-size", type=int, default=64, help="Database cache size in MB (default: 64)")
  parser.add_argument("--mmap-size", type=int, default=1024, help="Memory-mapped I/O size in MB (default: 1024)")
  parser.add_argument(
    "--check-versions", action="store_true", help="Only check and print database versions without merging"
  )
  parser.add_argument(
    "--dry-run", action="store_true", help="Preview merge statistics without creating output database"
  )

  args = parser.parse_args()

  try:
    merger = RclipDBMerger(
      args.db1,
      args.db2,
      args.output,
      verbose=args.verbose,
      batch_size=args.batch_size,
      commit_interval=args.commit_interval,
      cache_size_mb=args.cache_size,
      mmap_size_mb=args.mmap_size,
    )

    if args.check_versions:
      # Just check versions and exit
      merger.check_versions_only()
    else:
      # Check if output exists
      if Path(args.output).exists() and not args.force:
        print(f"Error: Output file '{args.output}' already exists. Use --force to overwrite.")
        sys.exit(1)
      merger.merge()
  except Exception as e:
    print(f"\nError: {e}")
    sys.exit(1)


if __name__ == "__main__":
  main()
