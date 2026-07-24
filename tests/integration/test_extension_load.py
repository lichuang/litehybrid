#!/usr/bin/env python3
"""Integration test that builds and loads the litehybrid SQLite extension."""

import sqlite3
import subprocess
import sys
from pathlib import Path


def project_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parents[2]


def build_extension() -> None:
    """Build the loadable extension in debug mode."""
    subprocess.run(
        ["cargo", "build", "-p", "litehybrid-ext", "--features", "extension"],
        cwd=project_root(),
        check=True,
    )


def extension_path() -> Path:
    """Return the platform-specific extension artifact path."""
    if sys.platform == "darwin":
        name = "liblitehybrid_ext.dylib"
    elif sys.platform == "linux":
        name = "liblitehybrid_ext.so"
    else:
        raise RuntimeError(f"unsupported platform: {sys.platform}")
    return project_root() / "target" / "debug" / name


def load_extension(conn: sqlite3.Connection) -> None:
    """Load the compiled litehybrid extension into a connection."""
    ext = extension_path()
    if not ext.exists():
        raise FileNotFoundError(f"extension artifact not found: {ext}")
    conn.enable_load_extension(True)
    conn.load_extension(str(ext))


def test_f32_vector_search(conn: sqlite3.Connection) -> None:
    """Smoke test for float32 vector search via the loaded extension."""
    conn.execute("CREATE VIRTUAL TABLE idx USING litehybrid(embedding float[3], metric='l2')")
    conn.execute("INSERT INTO idx(rowid, embedding) VALUES (1, vec_f32('[1.0, 0.0, 0.0]'))")
    conn.execute("INSERT INTO idx(rowid, embedding) VALUES (2, vec_f32('[0.0, 1.0, 0.0]'))")
    conn.execute("INSERT INTO idx(rowid, embedding) VALUES (3, vec_f32('[0.0, 0.0, 1.0]'))")

    rows = conn.execute(
        "SELECT rowid, distance FROM idx WHERE embedding = vec_f32('[1.0, 0.1, 0.1]') LIMIT 2"
    ).fetchall()
    assert len(rows) == 2, f"expected 2 rows, got {len(rows)}"
    assert rows[0][0] == 1, f"expected rowid 1 as nearest, got {rows[0][0]}"
    assert isinstance(rows[0][1], float), "distance should be a float"


def test_int8_vector_search(conn: sqlite3.Connection) -> None:
    """Smoke test for int8 vector search via the loaded extension."""
    conn.execute("CREATE VIRTUAL TABLE idx_i8 USING litehybrid(embedding int8[3], metric='l2')")
    conn.execute("INSERT INTO idx_i8(rowid, embedding) VALUES (1, vec_int8('[10, 0, 0]'))")
    conn.execute("INSERT INTO idx_i8(rowid, embedding) VALUES (2, vec_int8('[0, 10, 0]'))")
    conn.execute("INSERT INTO idx_i8(rowid, embedding) VALUES (3, vec_int8('[0, 0, 10]'))")

    rows = conn.execute(
        "SELECT rowid, distance FROM idx_i8 WHERE embedding = vec_int8('[10, 1, 1]') LIMIT 2"
    ).fetchall()
    assert len(rows) == 2, f"expected 2 rows, got {len(rows)}"
    assert rows[0][0] == 1, f"expected rowid 1 as nearest, got {rows[0][0]}"


def test_bit_vector_search(conn: sqlite3.Connection) -> None:
    """Smoke test for bit vector search via the loaded extension."""
    conn.execute("CREATE VIRTUAL TABLE idx_bit USING litehybrid(embedding bit[4])")
    conn.execute("INSERT INTO idx_bit(rowid, embedding) VALUES (1, vec_bit('[1, 0, 0, 0]'))")
    conn.execute("INSERT INTO idx_bit(rowid, embedding) VALUES (2, vec_bit('[0, 1, 0, 0]'))")
    conn.execute("INSERT INTO idx_bit(rowid, embedding) VALUES (3, vec_bit('[0, 0, 1, 0]'))")

    rows = conn.execute(
        "SELECT rowid, distance FROM idx_bit WHERE embedding = vec_bit('[1, 0, 1, 0]') LIMIT 2"
    ).fetchall()
    assert len(rows) == 2, f"expected 2 rows, got {len(rows)}"
    assert rows[0][0] == 1, f"expected rowid 1 as nearest, got {rows[0][0]}"


def test_dynamic_schema(conn: sqlite3.Connection) -> None:
    """Verify that vector + scalar metadata columns are declared correctly."""
    conn.execute(
        "CREATE VIRTUAL TABLE items USING litehybrid(embedding float[3], category text, year int)"
    )
    # table_xinfo includes hidden columns; the last field is the hidden flag.
    columns = conn.execute("PRAGMA table_xinfo(items)").fetchall()
    names = [row[1] for row in columns]
    types = {row[1]: row[2].upper() for row in columns}
    hidden = {row[1]: row[6] for row in columns}
    assert names == ["embedding", "category", "year", "distance", "k"], f"unexpected columns: {names}"
    assert types["embedding"] == "BLOB", f"unexpected embedding type: {types['embedding']}"
    assert types["category"] == "TEXT", f"unexpected category type: {types['category']}"
    assert types["year"] == "INT", f"unexpected year type: {types['year']}"
    assert hidden["distance"] == 1, "distance should be hidden"
    assert hidden["k"] == 1, "k should be hidden"


def main() -> int:
    """Run the integration tests."""
    build_extension()
    conn = sqlite3.connect(":memory:")
    load_extension(conn)

    test_f32_vector_search(conn)
    test_int8_vector_search(conn)
    test_bit_vector_search(conn)
    test_dynamic_schema(conn)

    print("all integration tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
