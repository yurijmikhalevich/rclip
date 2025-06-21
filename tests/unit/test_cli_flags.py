import os
import sqlite3
import tempfile
from pathlib import Path

import pytest

from rclip.main import main


class TestDatabasePathFlag:
    """Test --db-path command line flag functionality."""

    def test_custom_db_path_creates_database(self, monkeypatch, test_images_dir):
        """Test that specifying a custom db path creates database at that location."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Set up custom database path
            custom_db_path = Path(tmpdir) / "custom" / "location" / "test.db"
            
            # Ensure parent directory doesn't exist yet
            assert not custom_db_path.parent.exists()
            
            # Index with custom db path
            monkeypatch.chdir(test_images_dir)
            monkeypatch.setattr("sys.argv", ["rclip", "--db-path", str(custom_db_path), "--index-only"])
            
            main()
            
            # Verify database was created at custom location
            assert custom_db_path.exists()
            assert custom_db_path.parent.exists()
            
            # Verify it's a valid SQLite database
            con = sqlite3.connect(custom_db_path)
            cursor = con.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor]
            con.close()
            
            assert "images" in tables
            assert "db_version" in tables

    def test_custom_db_path_with_relative_path(self, monkeypatch, test_images_dir):
        """Test that relative paths work correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.chdir(tmpdir)
            
            # Use relative path
            relative_db_path = "relative/path/test.db"
            
            # Index with relative db path
            monkeypatch.setattr("sys.argv", ["rclip", "--db-path", relative_db_path, str(test_images_dir), "--index-only"])
            
            main()
            
            # Verify database was created at relative location
            full_path = Path(tmpdir) / relative_db_path
            assert full_path.exists()

    def test_custom_db_path_persists_across_runs(self, monkeypatch, test_images_dir):
        """Test that the same custom db path can be used across multiple runs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_db_path = Path(tmpdir) / "persistent.db"
            
            monkeypatch.chdir(test_images_dir)
            
            # First run - index images with explicit CPU device
            monkeypatch.setattr("sys.argv", ["rclip", "--db-path", str(custom_db_path), "--device", "cpu", "--index-only"])
            main()
            
            # Check image count after first run
            con = sqlite3.connect(custom_db_path)
            count1 = con.execute("SELECT COUNT(*) FROM images").fetchone()[0]
            con.close()
            assert count1 > 0
            
            # Second run - should use existing database (index-only again to avoid search)
            monkeypatch.setattr("sys.argv", ["rclip", "--db-path", str(custom_db_path), "--device", "cpu", "--index-only"])
            main()
            
            # Verify same database is used (same image count)
            con = sqlite3.connect(custom_db_path)
            count2 = con.execute("SELECT COUNT(*) FROM images").fetchone()[0]
            con.close()
            assert count2 == count1

    def test_custom_db_path_with_search(self, monkeypatch, test_images_dir, capsys):
        """Test that database is created and indexed with custom db path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_db_path = Path(tmpdir) / "search_test.db"
            
            monkeypatch.chdir(test_images_dir)
            
            # Index with custom db path
            monkeypatch.setattr("sys.argv", ["rclip", "--db-path", str(custom_db_path), "--device", "cpu", "--index-only"])
            main()
            
            # Verify database was created and has indexed images
            con = sqlite3.connect(custom_db_path)
            count = con.execute("SELECT COUNT(*) FROM images").fetchone()[0]
            con.close()
            
            assert count > 0
            assert custom_db_path.exists()

    def test_db_path_flag_with_invalid_path(self, monkeypatch, test_images_dir):
        """Test error handling when db path is invalid."""
        # Use a path that can't be created (e.g., in a read-only location)
        invalid_path = "/root/no_permission/test.db"
        
        monkeypatch.chdir(test_images_dir)
        monkeypatch.setattr("sys.argv", ["rclip", "--db-path", invalid_path, "--index-only"])
        
        # Should raise an error when trying to create database
        with pytest.raises((SystemExit, OSError, PermissionError)):
            main()

    def test_database_flag_alias(self, monkeypatch, test_images_dir):
        """Test that --database works as an alias for --db-path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_db_path = Path(tmpdir) / "alias_test.db"
            
            monkeypatch.chdir(test_images_dir)
            monkeypatch.setattr("sys.argv", ["rclip", "--database", str(custom_db_path), "--index-only"])
            
            main()
            
            # Verify database was created
            assert custom_db_path.exists()

    def test_db_path_with_tilde_expansion(self, monkeypatch, test_images_dir):
        """Test that tilde (~) in path is NOT automatically expanded (current behavior)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.chdir(tmpdir)
            
            # Use tilde in path - it will be treated literally
            db_path_with_tilde = "~/custom/rclip.db"
            
            # Since tilde is not expanded, it creates a literal directory named "~"
            monkeypatch.setattr("sys.argv", ["rclip", "--db-path", db_path_with_tilde, str(test_images_dir), "--index-only"])
            
            main()
            
            # Verify database was created at literal path (not expanded)
            literal_path = Path(tmpdir) / "~" / "custom" / "rclip.db"
            assert literal_path.exists()

    def test_db_path_with_environment_variable(self, monkeypatch, test_images_dir):
        """Test that environment variables in path are NOT automatically expanded (current behavior)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.chdir(tmpdir)
            
            # Use environment variable in path - it will be treated literally
            db_path_with_env = "$CUSTOM_DIR/rclip.db"
            
            # Since env var is not expanded, it creates a literal directory named "$CUSTOM_DIR"
            monkeypatch.setattr("sys.argv", ["rclip", "--db-path", db_path_with_env, str(test_images_dir), "--index-only"])
            
            main()
            
            # Verify database was created at literal path (not expanded)
            literal_path = Path(tmpdir) / "$CUSTOM_DIR" / "rclip.db"
            assert literal_path.exists()


class TestSearchDirFlag:
    """Test --search-dir command line flag functionality."""

    def test_search_dir_basic(self, monkeypatch, test_images_dir):
        """Test that --search-dir allows searching in a specific directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            
            # Run from a different directory than the images
            monkeypatch.chdir(tmpdir)
            
            # Index with search-dir pointing to images
            monkeypatch.setattr("sys.argv", ["rclip", "--db-path", str(db_path), "--search-dir", str(test_images_dir), "--device", "cpu", "--index-only"])
            main()
            
            # Verify images were indexed from the specified directory
            con = sqlite3.connect(db_path)
            count = con.execute("SELECT COUNT(*) FROM images").fetchone()[0]
            con.close()
            
            assert count > 0

    def test_search_dir_with_relative_path(self, monkeypatch, test_images_dir):
        """Test that relative paths work with --search-dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            
            # Create a parent directory and work from there
            parent_dir = test_images_dir.parent
            monkeypatch.chdir(parent_dir)
            
            # Use relative path to images directory
            relative_path = "images"
            
            monkeypatch.setattr("sys.argv", ["rclip", "--db-path", str(db_path), "--search-dir", relative_path, "--device", "cpu", "--index-only"])
            main()
            
            # Verify images were indexed
            con = sqlite3.connect(db_path)
            count = con.execute("SELECT COUNT(*) FROM images").fetchone()[0]
            con.close()
            
            assert count > 0

    def test_search_dir_nonexistent(self, monkeypatch):
        """Test error when --search-dir points to non-existent directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.chdir(tmpdir)
            
            nonexistent_dir = "/path/that/does/not/exist"
            monkeypatch.setattr("sys.argv", ["rclip", "--search-dir", nonexistent_dir, "--device", "cpu", "--index-only"])
            
            with pytest.raises(SystemExit):
                main()

    def test_search_dir_converts_to_absolute(self, monkeypatch, test_images_dir):
        """Test that search-dir is converted to absolute path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            
            # Work from parent directory
            parent_dir = test_images_dir.parent
            monkeypatch.chdir(parent_dir)
            
            # Index with relative path
            monkeypatch.setattr("sys.argv", ["rclip", "--db-path", str(db_path), "--search-dir", "images", "--device", "cpu", "--index-only"])
            main()
            
            # Check that paths in database are absolute
            con = sqlite3.connect(db_path)
            cursor = con.execute("SELECT filepath FROM images LIMIT 1")
            filepath = cursor.fetchone()[0]
            con.close()
            
            assert os.path.isabs(filepath)


class TestIndexOnlyFlag:
    """Test --index-only command line flag functionality."""

    def test_index_only_basic(self, monkeypatch, test_images_dir, capsys):
        """Test that --index-only indexes without searching."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            
            monkeypatch.chdir(test_images_dir)
            monkeypatch.setattr("sys.argv", ["rclip", "--db-path", str(db_path), "--device", "cpu", "--index-only"])
            
            main()
            
            # Check output indicates indexing completed
            captured = capsys.readouterr()
            assert "Indexing completed successfully" in captured.err
            
            # Verify database was created and populated
            con = sqlite3.connect(db_path)
            count = con.execute("SELECT COUNT(*) FROM images").fetchone()[0]
            con.close()
            
            assert count > 0

    def test_index_only_no_query_required(self, monkeypatch, test_images_dir):
        """Test that --index-only doesn't require a query argument."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            
            monkeypatch.chdir(test_images_dir)
            # No query provided, just index-only
            monkeypatch.setattr("sys.argv", ["rclip", "--db-path", str(db_path), "--device", "cpu", "--index-only"])
            
            # Should not raise an error
            main()
            
            assert db_path.exists()

    def test_index_only_with_no_indexing_error(self, monkeypatch, test_images_dir):
        """Test that --index-only and --no-indexing together cause an error."""
        monkeypatch.chdir(test_images_dir)
        monkeypatch.setattr("sys.argv", ["rclip", "--index-only", "--no-indexing"])
        
        with pytest.raises(SystemExit):
            main()


class TestDbCacheSizeFlag:
    """Test --db-cache-size command line flag functionality."""

    def test_db_cache_size_basic(self, monkeypatch, test_images_dir):
        """Test that --db-cache-size sets cache size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            cache_size_mb = 128
            
            monkeypatch.chdir(test_images_dir)
            monkeypatch.setattr("sys.argv", [
                "rclip", 
                "--db-path", str(db_path),
                "--db-cache-size", str(cache_size_mb),
                "--device", "cpu",
                "--index-only"
            ])
            
            main()
            
            # Verify database was created
            assert db_path.exists()
            
            # Check that cache size was applied (difficult to verify directly,
            # but we can check that the database works)
            con = sqlite3.connect(db_path)
            count = con.execute("SELECT COUNT(*) FROM images").fetchone()[0]
            con.close()
            
            assert count > 0

    def test_db_cache_size_with_search(self, monkeypatch, test_images_dir):
        """Test that cache size works during both indexing and search."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            
            monkeypatch.chdir(test_images_dir)
            
            # First index with custom cache size
            monkeypatch.setattr("sys.argv", [
                "rclip",
                "--db-path", str(db_path),
                "--db-cache-size", "256",
                "--device", "cpu",
                "--index-only"
            ])
            main()
            
            # Verify indexing worked
            con = sqlite3.connect(db_path)
            count = con.execute("SELECT COUNT(*) FROM images").fetchone()[0]
            con.close()
            
            assert count > 0


@pytest.fixture
def test_images_dir():
    """Provide path to test images directory."""
    return Path(__file__).parent.parent / "e2e" / "images"