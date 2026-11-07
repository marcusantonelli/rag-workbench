"""C# source via tree-sitter: a header block per type, one block per method-like member.

Type headers carry the XML doc comments, attributes, signature, fields,
short properties and a member list. Methods, constructors, operators,
indexers and long properties get their own blocks with their doc comments.
"""

from __future__ import annotations

from pathlib import Path

from rag.core.ingest.context import ExtractContext
from rag.core.types import Block, ExtractedDoc
from rag.core.util import read_text_file

_TYPE_NODES = {
    "class_declaration", "struct_declaration", "interface_declaration", "record_declaration",
    "record_struct_declaration",
}
_MEMBER_NODES = {
    "method_declaration", "constructor_declaration", "destructor_declaration", "operator_declaration",
    "conversion_operator_declaration", "indexer_declaration", "local_function_statement",
}
_INLINE_NODES = {"field_declaration", "event_field_declaration", "event_declaration", "delegate_declaration"}
_NAMESPACE_NODES = {"namespace_declaration", "file_scoped_namespace_declaration"}

_parser = None


def _get_parser():
    global _parser
    if _parser is None:
        import tree_sitter as ts
        import tree_sitter_c_sharp as cs

        _parser = ts.Parser(ts.Language(cs.language()))
    return _parser


class _Src:
    def __init__(self, data: bytes):
        self.data = data

    def text(self, node) -> str:
        return self.data[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    def span(self, start: int, end: int) -> str:
        return self.data[start:end].decode("utf-8", errors="replace")


def _name(node, src: _Src) -> str:
    name = node.child_by_field_name("name")
    if name is not None:
        return src.text(name)
    if node.type == "operator_declaration":
        op = node.child_by_field_name("operator")
        return f"operator {src.text(op)}" if op is not None else "operator"
    if node.type == "indexer_declaration":
        return "this[]"
    if node.type == "conversion_operator_declaration":
        typ = node.child_by_field_name("type")
        return f"operator {src.text(typ)}" if typ is not None else "operator"
    return node.type


def _doc_comments(node, src: _Src) -> tuple[str, int]:
    """Contiguous /// comments right above a node; returns (text, start_byte)."""
    comments = []
    start = node.start_byte
    prev = node.prev_sibling
    expected_row = node.start_point[0] - 1
    while prev is not None and prev.type == "comment" and prev.end_point[0] >= expected_row - 0:
        txt = src.text(prev)
        if not txt.startswith("///") and not txt.startswith("/**"):
            break
        comments.insert(0, txt)
        start = prev.start_byte
        expected_row = prev.start_point[0] - 1
        prev = prev.prev_sibling
    return ("\n".join(comments), start)


def _signature(node, src: _Src) -> str:
    body = node.child_by_field_name("body")
    if body is not None:
        return " ".join(src.span(node.start_byte, body.start_byte).split())
    return " ".join(src.text(node).split("\n", 1)[0].split())


def _is_long_property(node, src: _Src) -> bool:
    return node.end_point[0] - node.start_point[0] >= 6


def _line_of(src: _Src, byte: int) -> int:
    return src.data[:byte].count(b"\n") + 1


def _walk_type(node, prefix: str, src: _Src, blocks: list[Block]) -> None:
    name = _name(node, src)
    symbol = f"{prefix}.{name}" if prefix else name
    docs, start = _doc_comments(node, src)
    body = node.child_by_field_name("body")
    members_list: list[str] = []
    inline: list[str] = []
    children = list(body.named_children) if body is not None else []
    for child in children:
        if child.type in _MEMBER_NODES:
            members_list.append(_signature(child, src))
        elif child.type == "property_declaration":
            if _is_long_property(child, src):
                members_list.append(_signature(child, src))
            else:
                d, _ = _doc_comments(child, src)
                inline.append((d + "\n" if d else "") + src.text(child).strip())
        elif child.type in _INLINE_NODES:
            d, _ = _doc_comments(child, src)
            inline.append((d + "\n" if d else "") + src.text(child).strip())
        elif child.type in _TYPE_NODES or child.type == "enum_declaration":
            members_list.append(f"{child.type.replace('_declaration', '')} {_name(child, src)}")

    sig = _signature(node, src)
    parts = [p for p in (docs, sig + " {") if p]
    parts += ["    " + "\n    ".join(i.split("\n")) for i in inline]
    if members_list:
        parts.append("    // members: " + "; ".join(members_list))
    parts.append("}")
    blocks.append(Block(
        kind="code", text="```csharp\n" + "\n".join(parts) + "\n```", lang="csharp", symbol=symbol,
        line_start=_line_of(src, start), line_end=node.end_point[0] + 1,
        meta={"symbol_kind": node.type.replace("_declaration", ""), "signature": sig},
    ))
    for child in children:
        has_body = child.child_by_field_name("body") is not None
        if (child.type in _MEMBER_NODES and has_body) or (
                child.type == "property_declaration" and _is_long_property(child, src)):
            _add_member(child, symbol, src, blocks)
        elif child.type in _TYPE_NODES:
            _walk_type(child, symbol, src, blocks)
        elif child.type == "enum_declaration":
            _add_whole(child, symbol, src, blocks, "enum")


def _add_member(node, prefix: str, src: _Src, blocks: list[Block]) -> None:
    docs, start = _doc_comments(node, src)
    text = src.span(start, node.end_byte)
    blocks.append(Block(
        kind="code", text="```csharp\n" + text + "\n```", lang="csharp",
        symbol=f"{prefix}.{_name(node, src)}", line_start=_line_of(src, start), line_end=node.end_point[0] + 1,
        meta={"symbol_kind": node.type.replace("_declaration", ""), "signature": _signature(node, src)},
    ))


def _add_whole(node, prefix: str, src: _Src, blocks: list[Block], kind: str) -> None:
    docs, start = _doc_comments(node, src)
    name = _name(node, src)
    blocks.append(Block(
        kind="code", text="```csharp\n" + src.span(start, node.end_byte) + "\n```", lang="csharp",
        symbol=f"{prefix}.{name}" if prefix else name, line_start=_line_of(src, start),
        line_end=node.end_point[0] + 1, meta={"symbol_kind": kind, "signature": _signature(node, src)},
    ))


def _walk_container(node, prefix: str, src: _Src, blocks: list[Block], file_header: list[str]) -> None:
    for child in node.named_children:
        t = child.type
        if t in _NAMESPACE_NODES:
            ns = src.text(child.child_by_field_name("name"))
            full = f"{prefix}.{ns}" if prefix else ns
            file_header.append(f"namespace {full}")
            if t == "file_scoped_namespace_declaration":
                _walk_container(child, full, src, blocks, file_header)
            else:
                body = child.child_by_field_name("body")
                if body is not None:
                    _walk_container(body, full, src, blocks, file_header)
        elif t in _TYPE_NODES:
            _walk_type(child, prefix, src, blocks)
        elif t == "enum_declaration":
            _add_whole(child, prefix, src, blocks, "enum")
        elif t == "delegate_declaration":
            _add_whole(child, prefix, src, blocks, "delegate")
        elif t == "using_directive":
            file_header.append(src.text(child).strip())
        elif t == "global_statement":
            file_header.append(src.text(child).strip())


def _fallback(text: str, stem: str) -> list[Block]:
    blocks, buf, start = [], [], 1
    lines = text.split("\n")
    for i, line in enumerate(lines + [""], start=1):
        if line.strip():
            if not buf:
                start = i
            buf.append(line)
        elif buf:
            blocks.append(Block(kind="code", text="```csharp\n" + "\n".join(buf) + "\n```", lang="csharp",
                                symbol=stem, line_start=start, line_end=i - 1,
                                meta={"symbol_kind": "fragment", "signature": buf[0].strip()}))
            buf = []
    return blocks


def extract_csharp(path: Path, ctx: ExtractContext) -> ExtractedDoc:
    text, _ = read_text_file(path)
    data = text.encode("utf-8")
    tree = _get_parser().parse(data)
    src = _Src(data)
    blocks: list[Block] = []
    header: list[str] = []
    _walk_container(tree.root_node, "", src, blocks, header)
    warnings = []
    if tree.root_node.has_error:
        warnings.append("parse errors (preprocessor directives or invalid code); some members may be merged")
    if not blocks and not header:
        return ExtractedDoc(file_type="cs", title=None, blocks=_fallback(text, Path(path).stem),
                            extractor="csharp-fallback", warnings=warnings + ["no types found; split at blank lines"])
    if header:
        blocks.insert(0, Block(
            kind="code", text="```csharp\n" + "\n".join(header) + "\n```", lang="csharp",
            symbol=Path(path).stem, line_start=1, line_end=1,
            meta={"symbol_kind": "file", "signature": f"// file {Path(path).name}"},
        ))
    return ExtractedDoc(file_type="cs", title=Path(path).stem, blocks=blocks, extractor="csharp-tree-sitter",
                        warnings=warnings)
