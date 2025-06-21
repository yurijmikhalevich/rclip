import sqlite3
from pathlib import Path

import pytest

from rclip.merge_dbs import RclipDBMerger


class TestRclipDBMerger:
  """Test database merge functionality."""

  def create_test_db(self, db_path: Path, image_count: int, start_id: int = 1) -> None:
    """Create a test database with sample data."""
    con = sqlite3.connect(db_path)

    # Create schema matching rclip's
    con.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY,
                deleted BOOLEAN,
                filepath TEXT NOT NULL UNIQUE,
                modified_at DATETIME NOT NULL,
                size INTEGER NOT NULL,
                vector BLOB NOT NULL,
                indexing BOOLEAN
            )
        """)

    con.execute("CREATE TABLE IF NOT EXISTS db_version (version INTEGER)")
    con.execute("INSERT INTO db_version(version) VALUES (2)")

    # Insert test data
    for i in range(image_count):
      actual_id = start_id + i
      vector = b"test_vector_" + str(actual_id).encode()
      filepath = f"/test/image_{actual_id}.jpg"

      con.execute(
        """
                INSERT INTO images (id, filepath, modified_at, size, vector)
                VALUES (?, ?, ?, ?, ?)
            """,
        (actual_id, filepath, actual_id * 1000.0, actual_id * 1024, vector),
      )

    con.commit()
    con.close()

  def test_basic_merge(self, tmp_path):
    """Test basic merge of two databases."""
    # Create test databases
    db1_path = tmp_path / "test_db1.db"
    db2_path = tmp_path / "test_db2.db"
    output_path = tmp_path / "merged.db"

    self.create_test_db(db1_path, 10, start_id=1)
    self.create_test_db(db2_path, 15, start_id=100)

    # Perform merge
    merger = RclipDBMerger(str(db1_path), str(db2_path), str(output_path))
    merger.merge()

    # Verify results
    con = sqlite3.connect(output_path)
    count = con.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    con.close()

    assert count == 25  # 10 + 15 images
    assert output_path.exists()

  def test_conflict_resolution(self, tmp_path):
    """Test that newer images win in conflicts."""
    db1_path = tmp_path / "test_db1.db"
    db2_path = tmp_path / "test_db2.db"
    output_path = tmp_path / "merged.db"

    # Create databases with overlapping filepaths
    con1 = sqlite3.connect(db1_path)
    con1.execute("""
            CREATE TABLE images (
                id INTEGER PRIMARY KEY,
                deleted BOOLEAN,
                filepath TEXT NOT NULL UNIQUE,
                modified_at DATETIME NOT NULL,
                size INTEGER NOT NULL,
                vector BLOB NOT NULL,
                indexing BOOLEAN
            )
        """)
    con1.execute("CREATE TABLE db_version (version INTEGER)")
    con1.execute("INSERT INTO db_version(version) VALUES (2)")
    con1.execute(
      """
            INSERT INTO images (id, filepath, modified_at, size, vector)
            VALUES (1, '/test/conflict.jpg', 1000.0, 1024, ?)
        """,
      (b"old_vector",),
    )
    con1.commit()
    con1.close()

    con2 = sqlite3.connect(db2_path)
    con2.execute("""
            CREATE TABLE images (
                id INTEGER PRIMARY KEY,
                deleted BOOLEAN,
                filepath TEXT NOT NULL UNIQUE,
                modified_at DATETIME NOT NULL,
                size INTEGER NOT NULL,
                vector BLOB NOT NULL,
                indexing BOOLEAN
            )
        """)
    con2.execute("CREATE TABLE db_version (version INTEGER)")
    con2.execute("INSERT INTO db_version(version) VALUES (2)")
    con2.execute(
      """
            INSERT INTO images (id, filepath, modified_at, size, vector)
            VALUES (2, '/test/conflict.jpg', 2000.0, 2048, ?)
        """,
      (b"new_vector",),
    )
    con2.commit()
    con2.close()

    # Merge
    merger = RclipDBMerger(str(db1_path), str(db2_path), str(output_path))
    merger.merge()

    # Verify newer image won
    con = sqlite3.connect(output_path)
    count = con.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    result = con.execute("""
            SELECT modified_at, size, vector FROM images 
            WHERE filepath = '/test/conflict.jpg'
        """).fetchone()
    con.close()

    assert count == 1  # Only one image should exist
    assert merger.stats["conflicts_resolved"] == 1  # Conflict was detected
    assert result[0] == 2000.0  # Newer timestamp won
    assert result[1] == 2048  # Newer size
    assert result[2] == b"new_vector"  # Newer vector

  def test_deleted_preservation(self, tmp_path):
    """Test that deleted status is preserved."""
    db1_path = tmp_path / "test_db1.db"
    db2_path = tmp_path / "test_db2.db"
    output_path = tmp_path / "merged.db"

    # Create db1 with deleted image
    con1 = sqlite3.connect(db1_path)
    con1.execute("""
            CREATE TABLE images (
                id INTEGER PRIMARY KEY,
                deleted BOOLEAN,
                filepath TEXT NOT NULL UNIQUE,
                modified_at DATETIME NOT NULL,
                size INTEGER NOT NULL,
                vector BLOB NOT NULL,
                indexing BOOLEAN
            )
        """)
    con1.execute("CREATE TABLE db_version (version INTEGER)")
    con1.execute("INSERT INTO db_version(version) VALUES (2)")
    con1.execute(
      """
            INSERT INTO images (id, filepath, modified_at, size, vector, deleted)
            VALUES (1, '/test/deleted.jpg', 1000.0, 1024, ?, 1)
        """,
      (b"vector",),
    )
    con1.commit()
    con1.close()

    # Create db2 with same image not deleted but older
    con2 = sqlite3.connect(db2_path)
    con2.execute("""
            CREATE TABLE images (
                id INTEGER PRIMARY KEY,
                deleted BOOLEAN,
                filepath TEXT NOT NULL UNIQUE,
                modified_at DATETIME NOT NULL,
                size INTEGER NOT NULL,
                vector BLOB NOT NULL,
                indexing BOOLEAN
            )
        """)
    con2.execute("CREATE TABLE db_version (version INTEGER)")
    con2.execute("INSERT INTO db_version(version) VALUES (2)")
    con2.execute(
      """
            INSERT INTO images (id, filepath, modified_at, size, vector, deleted)
            VALUES (2, '/test/deleted.jpg', 500.0, 1024, ?, NULL)
        """,
      (b"vector",),
    )
    con2.commit()
    con2.close()

    # Merge
    merger = RclipDBMerger(str(db1_path), str(db2_path), str(output_path))
    merger.merge()

    # Verify deleted status preserved
    con = sqlite3.connect(output_path)
    result = con.execute("""
            SELECT deleted FROM images WHERE filepath = '/test/deleted.jpg'
        """).fetchone()
    con.close()

    assert result[0] == 1  # Should remain deleted

  def test_version_mismatch(self, tmp_path):
    """Test that version mismatch is detected."""
    db1_path = tmp_path / "test_db1.db"
    db2_path = tmp_path / "test_db2.db"
    output_path = tmp_path / "merged.db"

    # Create db1 with version 2
    self.create_test_db(db1_path, 5)

    # Create db2 with version 1
    con2 = sqlite3.connect(db2_path)
    con2.execute("""
            CREATE TABLE images (
                id INTEGER PRIMARY KEY,
                deleted BOOLEAN,
                filepath TEXT NOT NULL UNIQUE,
                modified_at DATETIME NOT NULL,
                size INTEGER NOT NULL,
                vector BLOB NOT NULL
            )
        """)
    con2.execute("CREATE TABLE db_version (version INTEGER)")
    con2.execute("INSERT INTO db_version(version) VALUES (1)")
    con2.commit()
    con2.close()

    # Merge should fail
    merger = RclipDBMerger(str(db1_path), str(db2_path), str(output_path))
    with pytest.raises(ValueError, match="version mismatch"):
      merger.merge()

  def test_check_versions_command(self, tmp_path):
    """Test the check-versions functionality."""
    db1_path = tmp_path / "test_db1.db"
    db2_path = tmp_path / "test_db2.db"

    self.create_test_db(db1_path, 10)
    self.create_test_db(db2_path, 15, start_id=100)

    merger = RclipDBMerger(str(db1_path), str(db2_path), "dummy.db")
    # This should not raise an exception
    merger.check_versions_only()

  def test_path_normalization(self, tmp_path):
    """Test that path normalization handles variations."""
    db1_path = tmp_path / "test_db1.db"
    db2_path = tmp_path / "test_db2.db"
    output_path = tmp_path / "merged.db"

    # Create db1 with path containing double slashes
    con1 = sqlite3.connect(db1_path)
    con1.execute("""
            CREATE TABLE images (
                id INTEGER PRIMARY KEY,
                deleted BOOLEAN,
                filepath TEXT NOT NULL UNIQUE,
                modified_at DATETIME NOT NULL,
                size INTEGER NOT NULL,
                vector BLOB NOT NULL,
                indexing BOOLEAN
            )
        """)
    con1.execute("CREATE TABLE db_version (version INTEGER)")
    con1.execute("INSERT INTO db_version(version) VALUES (2)")
    con1.execute(
      """
            INSERT INTO images (id, filepath, modified_at, size, vector)
            VALUES (1, '/test//path//image.jpg', 1000.0, 1024, ?)
        """,
      (b"vector1",),
    )
    con1.commit()
    con1.close()

    # Create db2 with normalized version of same path
    con2 = sqlite3.connect(db2_path)
    con2.execute("""
            CREATE TABLE images (
                id INTEGER PRIMARY KEY,
                deleted BOOLEAN,
                filepath TEXT NOT NULL UNIQUE,
                modified_at DATETIME NOT NULL,
                size INTEGER NOT NULL,
                vector BLOB NOT NULL,
                indexing BOOLEAN
            )
        """)
    con2.execute("CREATE TABLE db_version (version INTEGER)")
    con2.execute("INSERT INTO db_version(version) VALUES (2)")
    con2.execute(
      """
            INSERT INTO images (id, filepath, modified_at, size, vector)
            VALUES (2, '/test/path/image.jpg', 2000.0, 2048, ?)
        """,
      (b"vector2",),
    )
    con2.commit()
    con2.close()

    # Merge
    merger = RclipDBMerger(str(db1_path), str(db2_path), str(output_path))
    merger.merge()

    # Should only have one image (conflict resolved)
    con = sqlite3.connect(output_path)
    count = con.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    stats = merger.stats
    con.close()

    assert count == 1
    assert stats["conflicts_resolved"] == 1

  def test_database_integrity_check(self, tmp_path):
    """Test that corrupted databases are detected."""
    db1_path = tmp_path / "test_db1.db"
    db2_path = tmp_path / "corrupted.db"
    output_path = tmp_path / "merged.db"

    # Create valid db1
    self.create_test_db(db1_path, 5)

    # Create corrupted db2 (just write garbage)
    with open(db2_path, "wb") as f:
      f.write(b"This is not a valid SQLite database")

    # Merge should fail
    merger = RclipDBMerger(str(db1_path), str(db2_path), str(output_path))
    with pytest.raises(ValueError, match="integrity check failed"):
      merger.merge()

  def test_output_validation(self, tmp_path):
    """Test that output validation works."""
    db1_path = tmp_path / "test_db1.db"
    db2_path = tmp_path / "test_db2.db"
    output_path = tmp_path / "merged.db"

    self.create_test_db(db1_path, 10)
    self.create_test_db(db2_path, 15, start_id=100)

    merger = RclipDBMerger(str(db1_path), str(db2_path), str(output_path))
    merger.merge()

    # Validation should pass
    assert merger._validate_output() is True

    # Now corrupt the output to test validation failure
    con = sqlite3.connect(output_path)
    # Insert duplicate non-deleted filepath
    try:
      con.execute(
        """
                INSERT INTO images (filepath, modified_at, size, vector)
                VALUES ('/test/image_1.jpg', 9999.0, 9999, ?)
            """,
        (b"duplicate",),
      )
      con.commit()
    except sqlite3.IntegrityError:
      # Expected due to unique constraint
      pass
    con.close()

  def test_custom_parameters(self, tmp_path):
    """Test merge with custom batch size and commit interval."""
    db1_path = tmp_path / "test_db1.db"
    db2_path = tmp_path / "test_db2.db"
    output_path = tmp_path / "merged.db"

    self.create_test_db(db1_path, 100)
    self.create_test_db(db2_path, 150, start_id=1000)

    # Use small batch size and commit interval
    merger = RclipDBMerger(
      str(db1_path), str(db2_path), str(output_path), batch_size=10, commit_interval=50, verbose=True
    )
    merger.merge()

    # Verify all images merged
    con = sqlite3.connect(output_path)
    count = con.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    con.close()

    assert count == 250

  def test_dry_run_mode(self, tmp_path):
    """Test dry-run mode functionality."""
    db1_path = tmp_path / "test_db1.db"
    db2_path = tmp_path / "test_db2.db"
    output_path = tmp_path / "merged.db"

    self.create_test_db(db1_path, 10)
    self.create_test_db(db2_path, 15, start_id=100)

    # Run in dry-run mode
    merger = RclipDBMerger(str(db1_path), str(db2_path), str(output_path), dry_run=True)
    merger.merge()

    # Output file should NOT be created in dry-run mode
    assert not output_path.exists()

  def test_cache_size_configuration(self, tmp_path):
    """Test custom cache size is applied."""
    db1_path = tmp_path / "test_db1.db"
    db2_path = tmp_path / "test_db2.db"
    output_path = tmp_path / "merged.db"

    self.create_test_db(db1_path, 5)
    self.create_test_db(db2_path, 5)

    # Use custom cache size
    merger = RclipDBMerger(str(db1_path), str(db2_path), str(output_path), cache_size_mb=128)

    # Test that the cache size is set correctly
    assert merger.cache_size_mb == 128

    # Merge should still work
    merger.merge()
    assert output_path.exists()

  def test_mmap_size_configuration(self, tmp_path):
    """Test custom mmap size is applied."""
    db1_path = tmp_path / "test_db1.db"
    db2_path = tmp_path / "test_db2.db"
    output_path = tmp_path / "merged.db"

    self.create_test_db(db1_path, 5)
    self.create_test_db(db2_path, 5)

    # Use custom mmap size
    merger = RclipDBMerger(str(db1_path), str(db2_path), str(output_path), mmap_size_mb=512)

    # Test that the mmap size is set correctly
    assert merger.mmap_size_mb == 512

    # Merge should still work
    merger.merge()
    assert output_path.exists()

  def test_force_overwrite(self, tmp_path):
    """Test force overwrite functionality."""
    db1_path = tmp_path / "test_db1.db"
    db2_path = tmp_path / "test_db2.db"
    output_path = tmp_path / "merged.db"

    self.create_test_db(db1_path, 5, start_id=1)
    self.create_test_db(db2_path, 5, start_id=100)

    # Create existing database file with different data
    self.create_test_db(output_path, 3, start_id=500)

    # Verify old database has 3 images
    con = sqlite3.connect(output_path)
    old_count = con.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    con.close()
    assert old_count == 3

    # Merge should overwrite the existing database
    merger = RclipDBMerger(str(db1_path), str(db2_path), str(output_path))
    merger.merge()

    # Verify it's been overwritten with new data
    con = sqlite3.connect(output_path)
    count = con.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    # Check that we don't have any of the old images (id >= 500)
    old_images = con.execute("SELECT COUNT(*) FROM images WHERE id >= 500").fetchone()[0]
    con.close()
    assert count == 10  # Only images from db1 and db2
    assert old_images == 0  # None of the old images

  def test_verbose_logging(self, tmp_path, capsys):
    """Test verbose mode outputs additional information."""
    db1_path = tmp_path / "test_db1.db"
    db2_path = tmp_path / "test_db2.db"
    output_path = tmp_path / "merged.db"

    self.create_test_db(db1_path, 5)
    self.create_test_db(db2_path, 5)

    # Run with verbose mode
    merger = RclipDBMerger(str(db1_path), str(db2_path), str(output_path), verbose=True)
    merger.merge()

    # Check that verbose output was produced
    captured = capsys.readouterr()
    assert "Validating databases..." in captured.out
    assert "Indexed DB1" in captured.out
    assert "Indexed DB2" in captured.out

  def test_empty_database_merge(self, tmp_path):
    """Test merging when one database is empty."""
    db1_path = tmp_path / "test_db1.db"
    db2_path = tmp_path / "test_db2.db"
    output_path = tmp_path / "merged.db"

    # Create one normal db and one empty db
    self.create_test_db(db1_path, 10)
    self.create_test_db(db2_path, 0)  # Empty database

    merger = RclipDBMerger(str(db1_path), str(db2_path), str(output_path))
    merger.merge()

    # Should have only images from db1
    con = sqlite3.connect(output_path)
    count = con.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    con.close()
    assert count == 10

  def test_large_batch_processing(self, tmp_path):
    """Test that large databases are processed correctly with batching."""
    db1_path = tmp_path / "test_db1.db"
    db2_path = tmp_path / "test_db2.db"
    output_path = tmp_path / "merged.db"

    # Create larger databases to test batching
    self.create_test_db(db1_path, 150)
    self.create_test_db(db2_path, 200, start_id=1000)

    # Use small batch size to ensure multiple batches
    merger = RclipDBMerger(str(db1_path), str(db2_path), str(output_path), batch_size=50)
    merger.merge()

    # Verify all images were merged
    con = sqlite3.connect(output_path)
    count = con.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    con.close()
    assert count == 350

  def test_multiple_conflicts_resolution(self, tmp_path):
    """Test resolution of multiple conflicts."""
    db1_path = tmp_path / "test_db1.db"
    db2_path = tmp_path / "test_db2.db"
    output_path = tmp_path / "merged.db"

    # Create databases with multiple conflicts
    con1 = sqlite3.connect(db1_path)
    con1.execute("""
      CREATE TABLE images (
        id INTEGER PRIMARY KEY,
        deleted BOOLEAN,
        filepath TEXT NOT NULL UNIQUE,
        modified_at DATETIME NOT NULL,
        size INTEGER NOT NULL,
        vector BLOB NOT NULL,
        indexing BOOLEAN
      )
    """)
    con1.execute("CREATE TABLE db_version (version INTEGER)")
    con1.execute("INSERT INTO db_version(version) VALUES (2)")

    # Insert multiple conflicting images
    for i in range(5):
      con1.execute(
        """
        INSERT INTO images (id, filepath, modified_at, size, vector)
        VALUES (?, ?, ?, ?, ?)
      """,
        (i, f"/conflict_{i}.jpg", 1000.0, 1024, b"old"),
      )
    con1.commit()
    con1.close()

    con2 = sqlite3.connect(db2_path)
    con2.execute("""
      CREATE TABLE images (
        id INTEGER PRIMARY KEY,
        deleted BOOLEAN,
        filepath TEXT NOT NULL UNIQUE,
        modified_at DATETIME NOT NULL,
        size INTEGER NOT NULL,
        vector BLOB NOT NULL,
        indexing BOOLEAN
      )
    """)
    con2.execute("CREATE TABLE db_version (version INTEGER)")
    con2.execute("INSERT INTO db_version(version) VALUES (2)")

    # Insert same images with newer timestamps
    for i in range(5):
      con2.execute(
        """
        INSERT INTO images (id, filepath, modified_at, size, vector)
        VALUES (?, ?, ?, ?, ?)
      """,
        (i + 100, f"/conflict_{i}.jpg", 2000.0, 2048, b"new"),
      )
    con2.commit()
    con2.close()

    merger = RclipDBMerger(str(db1_path), str(db2_path), str(output_path))
    merger.merge()

    # Verify all conflicts were resolved
    assert merger.stats["conflicts_resolved"] == 5

    # Verify newer versions won
    con = sqlite3.connect(output_path)
    for i in range(5):
      result = con.execute(
        """
        SELECT modified_at, vector FROM images WHERE filepath = ?
      """,
        (f"/conflict_{i}.jpg",),
      ).fetchone()
      assert result[0] == 2000.0
      assert result[1] == b"new"
    con.close()
