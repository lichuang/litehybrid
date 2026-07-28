#!/usr/bin/env python3
"""Integration tests that build and load the litehybrid SQLite extension."""

import sqlite3
import subprocess
import sys
import unittest
from pathlib import Path


class ExtensionLoadTestCase(unittest.TestCase):
    """Tests for the litehybrid loadable SQLite extension."""

    _extension_path: Path

    @classmethod
    def setUpClass(cls) -> None:
        """Build the loadable extension once before running all tests."""
        cls._build_extension()
        cls._extension_path = cls._extension_path_for_current_platform()
        if not cls._extension_path.exists():
            raise FileNotFoundError(f"extension artifact not found: {cls._extension_path}")

    def setUp(self) -> None:
        """Create a fresh in-memory SQLite connection with the extension loaded."""
        print(f"\n>>> Running {self.id()}", flush=True)
        self.conn = sqlite3.connect(":memory:")
        self.conn.enable_load_extension(True)
        self.conn.load_extension(str(self._extension_path))
        self.addCleanup(self.conn.close)

    @staticmethod
    def _project_root() -> Path:
        return Path(__file__).resolve().parents[2]

    @classmethod
    def _build_extension(cls) -> None:
        subprocess.run(
            ["cargo", "build", "-p", "litehybrid-ext", "--features", "extension"],
            cwd=cls._project_root(),
            check=True,
        )

    @classmethod
    def _extension_path_for_current_platform(cls) -> Path:
        if sys.platform == "darwin":
            name = "liblitehybrid_ext.dylib"
        elif sys.platform == "linux":
            name = "liblitehybrid_ext.so"
        else:
            raise RuntimeError(f"unsupported platform: {sys.platform}")
        return cls._project_root() / "target" / "debug" / name

    def test_f32_vector_search(self) -> None:
        """Smoke test for float32 vector search via the loaded extension."""
        self.conn.execute("CREATE VIRTUAL TABLE idx USING litehybrid(embedding float[3], metric='l2')")
        self.conn.execute("INSERT INTO idx(rowid, embedding) VALUES (1, vec_f32('[1.0, 0.0, 0.0]'))")
        self.conn.execute("INSERT INTO idx(rowid, embedding) VALUES (2, vec_f32('[0.0, 1.0, 0.0]'))")
        self.conn.execute("INSERT INTO idx(rowid, embedding) VALUES (3, vec_f32('[0.0, 0.0, 1.0]'))")

        rows = self.conn.execute(
            "SELECT rowid, distance FROM idx WHERE embedding = vec_f32('[1.0, 0.1, 0.1]') LIMIT 2"
        ).fetchall()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][0], 1)
        self.assertIsInstance(rows[0][1], float)

    def test_int8_vector_search(self) -> None:
        """Smoke test for int8 vector search via the loaded extension."""
        self.conn.execute("CREATE VIRTUAL TABLE idx_i8 USING litehybrid(embedding int8[3], metric='l2')")
        self.conn.execute("INSERT INTO idx_i8(rowid, embedding) VALUES (1, vec_int8('[10, 0, 0]'))")
        self.conn.execute("INSERT INTO idx_i8(rowid, embedding) VALUES (2, vec_int8('[0, 10, 0]'))")
        self.conn.execute("INSERT INTO idx_i8(rowid, embedding) VALUES (3, vec_int8('[0, 0, 10]'))")

        rows = self.conn.execute(
            "SELECT rowid, distance FROM idx_i8 WHERE embedding = vec_int8('[10, 1, 1]') LIMIT 2"
        ).fetchall()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][0], 1)

    def test_bit_vector_search(self) -> None:
        """Smoke test for bit vector search via the loaded extension."""
        self.conn.execute("CREATE VIRTUAL TABLE idx_bit USING litehybrid(embedding bit[4])")
        self.conn.execute("INSERT INTO idx_bit(rowid, embedding) VALUES (1, vec_bit('[1, 0, 0, 0]'))")
        self.conn.execute("INSERT INTO idx_bit(rowid, embedding) VALUES (2, vec_bit('[0, 1, 0, 0]'))")
        self.conn.execute("INSERT INTO idx_bit(rowid, embedding) VALUES (3, vec_bit('[0, 0, 1, 0]'))")

        rows = self.conn.execute(
            "SELECT rowid, distance FROM idx_bit WHERE embedding = vec_bit('[1, 0, 1, 0]') LIMIT 2"
        ).fetchall()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][0], 1)

    def test_dynamic_schema(self) -> None:
        """Verify that vector + scalar metadata columns are declared correctly."""
        self.conn.execute(
            "CREATE VIRTUAL TABLE items_schema USING litehybrid(embedding float[3], category text, year int)"
        )
        # table_xinfo includes hidden columns; the last field is the hidden flag.
        columns = self.conn.execute("PRAGMA table_xinfo(items_schema)").fetchall()
        names = [row[1] for row in columns]
        types = {row[1]: row[2].upper() for row in columns}
        hidden = {row[1]: row[6] for row in columns}

        self.assertEqual(names, ["embedding", "category", "year", "distance", "k"])
        self.assertEqual(types["embedding"], "BLOB")
        self.assertEqual(types["category"], "TEXT")
        self.assertEqual(types["year"], "INT")
        self.assertEqual(hidden["distance"], 1)
        self.assertEqual(hidden["k"], 1)

    def test_metadata_roundtrip(self) -> None:
        """Verify that scalar metadata columns are persisted and readable via the virtual table."""
        self.conn.execute(
            "CREATE VIRTUAL TABLE items_meta USING litehybrid(embedding float[3], category text, year int)"
        )
        self.conn.execute(
            "INSERT INTO items_meta(rowid, category, year, embedding) VALUES (1, 'tech', 2024, vec_f32('[1.0, 0.0, 0.0]'))"
        )
        self.conn.execute(
            "INSERT INTO items_meta(rowid, category, year, embedding) VALUES (2, 'science', 2023, vec_f32('[0.0, 1.0, 0.0]'))"
        )

        # Read metadata back through the virtual table (requires a vector constraint).
        rows = self.conn.execute(
            "SELECT rowid, category, year FROM items_meta WHERE embedding = vec_f32('[1.0, 0.0, 0.0]') AND k = 1 ORDER BY distance"
        ).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0], (1, "tech", 2024))

        # Also inspect the shadow table directly to confirm the values were stored.
        rows = self.conn.execute(
            "SELECT rowid, category, year FROM items_meta_litehybrid_metadata ORDER BY rowid"
        ).fetchall()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0], (1, "tech", 2024))
        self.assertEqual(rows[1], (2, "science", 2023))

    def test_metadata_type_mismatch(self) -> None:
        """Verify that inserting a text value into an integer metadata column fails."""
        self.conn.execute(
            "CREATE VIRTUAL TABLE items_types USING litehybrid(embedding float[3], year int)"
        )
        with self.assertRaises(sqlite3.OperationalError) as ctx:
            self.conn.execute(
                "INSERT INTO items_types(rowid, year, embedding) VALUES (1, 'not-a-number', vec_f32('[1.0, 0.0, 0.0]'))"
            )
        self.assertIn("metadata type mismatch", str(ctx.exception).lower())

    def test_metadata_constraint_filters_results(self) -> None:
        """Verify that metadata column constraints are applied during the vector search."""
        self.conn.execute(
            "CREATE VIRTUAL TABLE items_filter USING litehybrid(embedding float[3], category text, year int)"
        )
        self.conn.execute(
            "INSERT INTO items_filter(rowid, category, year, embedding) VALUES (1, 'tech', 2024, vec_f32('[1.0, 0.0, 0.0]'))"
        )
        self.conn.execute(
            "INSERT INTO items_filter(rowid, category, year, embedding) VALUES (2, 'science', 2023, vec_f32('[0.0, 1.0, 0.0]'))"
        )
        self.conn.execute(
            "INSERT INTO items_filter(rowid, category, year, embedding) VALUES (3, 'tech', 2022, vec_f32('[0.9, 0.0, 0.0]'))"
        )

        # Single metadata constraint, also reading the metadata column back.
        rows = self.conn.execute(
            "SELECT rowid, category FROM items_filter WHERE embedding = vec_f32('[1.0, 0.1, 0.1]') "
            "AND category = 'tech' ORDER BY distance"
        ).fetchall()
        self.assertEqual(rows, [(1, "tech"), (3, "tech")])

        # Multiple metadata constraints plus explicit k.
        rows = self.conn.execute(
            "SELECT rowid, category, year FROM items_filter WHERE embedding = vec_f32('[1.0, 0.1, 0.1]') "
            "AND category = 'tech' AND year > 2020 AND k = 5 ORDER BY distance"
        ).fetchall()
        self.assertEqual(rows, [(1, "tech", 2024), (3, "tech", 2022)])

        # Constraint that excludes all matching rows.
        rows = self.conn.execute(
            "SELECT rowid FROM items_filter WHERE embedding = vec_f32('[1.0, 0.1, 0.1]') "
            "AND category = 'science' AND year > 2025 ORDER BY distance"
        ).fetchall()
        self.assertEqual(rows, [])

        # Vector column is not the first declared column.
        self.conn.execute(
            "CREATE VIRTUAL TABLE items_filter2 USING litehybrid(category text, embedding float[3], year int)"
        )
        self.conn.execute(
            "INSERT INTO items_filter2(rowid, category, year, embedding) VALUES (1, 'tech', 2024, vec_f32('[1.0, 0.0, 0.0]'))"
        )
        self.conn.execute(
            "INSERT INTO items_filter2(rowid, category, year, embedding) VALUES (2, 'science', 2023, vec_f32('[0.0, 1.0, 0.0]'))"
        )
        self.conn.execute(
            "INSERT INTO items_filter2(rowid, category, year, embedding) VALUES (3, 'tech', 2022, vec_f32('[0.9, 0.0, 0.0]'))"
        )
        rows = self.conn.execute(
            "SELECT rowid, category, year FROM items_filter2 WHERE embedding = vec_f32('[1.0, 0.1, 0.1]') "
            "AND category = 'tech' AND year > 2020 ORDER BY distance"
        ).fetchall()
        self.assertEqual(rows, [(1, "tech", 2024), (3, "tech", 2022)])

    def test_update_metadata(self) -> None:
        """Verify that updating metadata columns without changing the vector works."""
        self.conn.execute(
            "CREATE VIRTUAL TABLE items_update USING litehybrid(embedding float[3], category text, year int)"
        )
        self.conn.execute(
            "INSERT INTO items_update(rowid, category, year, embedding) VALUES (1, 'tech', 2024, vec_f32('[1.0, 0.0, 0.0]'))"
        )

        # Update only metadata columns; the vector stays the same.
        self.conn.execute(
            "UPDATE items_update SET category = 'science', year = 2025 WHERE rowid = 1"
        )

        rows = self.conn.execute(
            "SELECT rowid, category, year FROM items_update WHERE embedding = vec_f32('[1.0, 0.0, 0.0]') "
            "AND category = 'science' AND k = 1 ORDER BY distance"
        ).fetchall()
        self.assertEqual(rows, [(1, "science", 2025)])

    def test_delete_row(self) -> None:
        """Verify that deleting a row removes it from metadata-filtered queries."""
        self.conn.execute(
            "CREATE VIRTUAL TABLE items_delete USING litehybrid(embedding float[3], category text, year int)"
        )
        self.conn.execute(
            "INSERT INTO items_delete(rowid, category, year, embedding) VALUES (1, 'tech', 2024, vec_f32('[1.0, 0.0, 0.0]'))"
        )
        self.conn.execute(
            "INSERT INTO items_delete(rowid, category, year, embedding) VALUES (2, 'science', 2023, vec_f32('[0.0, 1.0, 0.0]'))"
        )

        self.conn.execute("DELETE FROM items_delete WHERE rowid = 1")

        rows = self.conn.execute(
            "SELECT rowid FROM items_delete WHERE embedding = vec_f32('[1.0, 0.0, 0.0]') "
            "AND category = 'tech' AND k = 1 ORDER BY distance"
        ).fetchall()
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
