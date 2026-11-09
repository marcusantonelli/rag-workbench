# PyInstaller spec for the RAG Workbench engine (one folder, console app with no window on Windows).
# Build with:  python desktop/scripts/build_engine.py
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

hidden = (
    collect_submodules("rag")
    + collect_submodules("rapidocr_onnxruntime")
    + collect_submodules("sqlglot")
    + ["tree_sitter", "tree_sitter_c_sharp", "sqlite_vec", "pypdfium2", "pypdfium2_raw", "pdfplumber",
       "docx", "openpyxl", "bs4", "lxml", "charset_normalizer", "tokenizers"]
)
datas = (
    collect_data_files("rag")
    + collect_data_files("rapidocr_onnxruntime")
    + collect_data_files("sqlite_vec")
    + collect_data_files("pypdfium2_raw")
    + collect_data_files("docx")
)
binaries = collect_dynamic_libs("pypdfium2_raw") + collect_dynamic_libs("onnxruntime")

# sqlite-vec ships its loadable extension as a bare vec0.so / vec0.dll / vec0.dylib next to
# its __init__.py, which the library hooks don't recognise; add it where sqlite_vec.load() looks.
import glob
import os

import sqlite_vec

_vec = [p for p in glob.glob(sqlite_vec.loadable_path() + ".*") if p.endswith((".so", ".dll", ".dylib"))]
if not _vec:
    raise SystemExit("sqlite-vec's vec0 library was not found; reinstall sqlite-vec")
binaries += [(p, "sqlite_vec") for p in _vec]

EXCLUDES = ["tkinter", "matplotlib", "IPython", "pytest", "reportlab", "torch", "docling", "transformers",
            "pysqlite3", "sqlean", "llama_cpp", "qwen3_embed"]

# Two programs sharing one folder of libraries: the app's engine and the `rag` command.
engine = Analysis(["entry.py"], binaries=binaries, datas=datas, hiddenimports=hidden, excludes=EXCLUDES)
cli = Analysis(["cli_entry.py"], binaries=binaries, datas=datas, hiddenimports=hidden, excludes=EXCLUDES)

engine_exe = EXE(PYZ(engine.pure), engine.scripts, [], exclude_binaries=True, name="rag-engine", console=True,
                 upx=False, strip=False, debug=False)
cli_exe = EXE(PYZ(cli.pure), cli.scripts, [], exclude_binaries=True, name="rag", console=True,
              upx=False, strip=False, debug=False)
coll = COLLECT(engine_exe, cli_exe, engine.binaries, engine.datas, cli.binaries, cli.datas,
               name="rag-engine", upx=False, strip=False)
