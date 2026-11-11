"""RAG Builder: turns a mixed document set into a portable SQLite pack.

The pack is two SQLite files (``rag.db`` with chunks, FTS5 and sqlite-vec
vectors; ``tables.db`` with spreadsheet data as real tables) plus figure crops
and a manifest. A small Python retrieval path is included for testing packs
before the C# retrieval server exists.
"""

__version__ = "0.4.3"
SCHEMA_VERSION = 1
PACK_FORMAT = "ragpack/1"
# ragpack/2 is ragpack/1 with the vectors in vectors.shammy instead of rag.db's chunk_vec table.
PACK_FORMAT_SHAMMY = "ragpack/2"
PACK_FORMATS = (PACK_FORMAT, PACK_FORMAT_SHAMMY)
