"""Python source: one block per function/method, a header block per class and module."""

from __future__ import annotations

import ast
from pathlib import Path

from rag.core.ingest.context import ExtractContext
from rag.core.types import Block, ExtractedDoc
from rag.core.util import read_text_file


def _start_line(node: ast.AST) -> int:
    decos = getattr(node, "decorator_list", None) or []
    return min([d.lineno for d in decos] + [node.lineno])


def _src(lines: list[str], start: int, end: int) -> str:
    return "\n".join(lines[start - 1:end]).rstrip()


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef, lines: list[str]) -> str:
    body_start = node.body[0].lineno if node.body else node.lineno
    sig_end = max(node.lineno, body_start - 1)
    if (node.body and isinstance(node.body[0], ast.Expr)
            and isinstance(getattr(node.body[0], "value", None), ast.Constant)
            and isinstance(node.body[0].value.value, str)):
        sig_end = node.body[0].end_lineno or sig_end
    return _src(lines, _start_line(node), sig_end)


def _member_summary(node: ast.AST) -> str:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = [a.arg for a in node.args.args]
        return f"{node.name}({', '.join(args)})"
    if isinstance(node, ast.ClassDef):
        return f"class {node.name}"
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        return ", ".join(ast.unparse(t) for t in targets)
    return ""


def _walk_class(node: ast.ClassDef, prefix: str, lines: list[str], blocks: list[Block]) -> None:
    symbol = f"{prefix}.{node.name}"
    members = [m for m in (_member_summary(n) for n in node.body) if m]
    header_parts = [_signature(node, lines)]
    for child in node.body:
        if isinstance(child, (ast.Assign, ast.AnnAssign)):
            header_parts.append(_src(lines, child.lineno, child.end_lineno or child.lineno))
    if members:
        header_parts.append("# members: " + "; ".join(members))
    blocks.append(Block(
        kind="code", text="```python\n" + "\n".join(header_parts) + "\n```", lang="python",
        symbol=symbol, line_start=_start_line(node), line_end=node.end_lineno,
        meta={"symbol_kind": "class", "signature": _signature(node, lines).split("\n", 1)[0]},
    ))
    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _add_function(child, symbol, lines, blocks)
        elif isinstance(child, ast.ClassDef):
            _walk_class(child, symbol, lines, blocks)


def _add_function(node: ast.FunctionDef | ast.AsyncFunctionDef, prefix: str, lines: list[str], blocks: list[Block]) -> None:
    start, end = _start_line(node), node.end_lineno or node.lineno
    blocks.append(Block(
        kind="code", text="```python\n" + _src(lines, start, end) + "\n```", lang="python",
        symbol=f"{prefix}.{node.name}", line_start=start, line_end=end,
        meta={"symbol_kind": "function", "signature": _signature(node, lines)},
    ))


def _fallback(text: str, module: str) -> list[Block]:
    blocks, buf, start = [], [], 1
    for i, line in enumerate(text.split("\n") + [""], start=1):
        if line.strip():
            if not buf:
                start = i
            buf.append(line)
        elif buf and (len(buf) > 3 or i == len(text.split("\n")) + 1):
            blocks.append(Block(kind="code", text="```python\n" + "\n".join(buf) + "\n```", lang="python",
                                symbol=module, line_start=start, line_end=i - 1,
                                meta={"symbol_kind": "fragment", "signature": buf[0]}))
            buf = []
    return blocks


def extract_python(path: Path, ctx: ExtractContext) -> ExtractedDoc:
    text, _ = read_text_file(path)
    module = Path(path).stem
    lines = text.split("\n")
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return ExtractedDoc(file_type="py", title=None, blocks=_fallback(text, module), extractor="python-fallback",
                            warnings=[f"syntax error at line {exc.lineno}; split at blank lines"])
    blocks: list[Block] = []
    header: list[str] = []
    header_lines: list[int] = []
    doc = ast.get_docstring(tree)
    if doc:
        header.append(f'"""{doc}"""')
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _add_function(node, module, lines, blocks)
        elif isinstance(node, ast.ClassDef):
            _walk_class(node, module, lines, blocks)
        elif isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant) and node is tree.body[0]:
            header_lines.append(node.lineno)
            continue
        else:
            seg = _src(lines, node.lineno, node.end_lineno or node.lineno)
            header.append(seg)
            header_lines.extend([node.lineno, node.end_lineno or node.lineno])
    if header:
        blocks.insert(0, Block(
            kind="code", text="```python\n" + "\n\n".join(header) + "\n```", lang="python",
            symbol=module, line_start=min(header_lines) if header_lines else 1,
            line_end=max(header_lines) if header_lines else 1,
            meta={"symbol_kind": "module", "signature": f"# module {module}"},
        ))
    return ExtractedDoc(file_type="py", title=module, blocks=blocks, extractor="python-ast")
