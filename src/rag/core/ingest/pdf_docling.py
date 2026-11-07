"""Docling PDF backend (optional: `pip install rag-builder[docling]`).

Not exercised in the development sandbox (Docling's layout models download from
Hugging Face on first use). On first real use, run `docling-tools models
download` once, or let it fetch, then set pdf.backend = "docling" to make
failures loud instead of falling back to the native extractor.
"""

from __future__ import annotations

from pathlib import Path

from rag.core.ingest.context import ExtractContext
from rag.core.types import Block, ExtractedDoc
from rag.core.util import long_path, markdown_table

_converter = None


def _get_converter(ctx: ExtractContext):
    global _converter
    if _converter is not None:
        return _converter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions()
    opts.do_ocr = True
    opts.do_table_structure = True
    opts.table_structure_options.mode = TableFormerMode.ACCURATE
    opts.generate_picture_images = True
    opts.images_scale = 2.0
    try:  # RapidOCR keeps OCR on ONNX Runtime
        from docling.datamodel.pipeline_options import RapidOcrOptions

        opts.ocr_options = RapidOcrOptions()
    except ImportError:
        pass
    try:  # broken text layers (unmapped fonts) get re-read with OCR
        opts.ocr_options.replace_broken_text_layer = True
    except (AttributeError, ValueError):
        pass
    try:  # PDF headings otherwise all come out as level 1
        from docling.datamodel.pipeline_options import HeadingHierarchyOptions

        opts.heading_hierarchy_options = HeadingHierarchyOptions(enabled=True)
    except (ImportError, AttributeError, ValueError):
        pass
    try:  # labels logos, icons, stamps so they can be skipped
        opts.do_picture_classification = True
    except (AttributeError, ValueError):
        pass
    _converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    return _converter


_SKIP_PICTURE_LABELS = {"logo", "icon", "stamp", "signature", "bar_code", "qr_code"}


def _picture_label(item) -> str | None:
    for attr in ("meta", "annotations"):
        obj = getattr(item, attr, None)
        if obj is None:
            continue
        classification = getattr(obj, "classification", None)
        preds = getattr(classification, "predictions", None) if classification else None
        if preds:
            return str(getattr(preds[0], "class_name", "")).lower() or None
        if isinstance(obj, list):
            for ann in obj:
                classes = getattr(ann, "predicted_classes", None)
                if classes:
                    return str(classes[0].class_name).lower()
    return None


def extract_pdf_docling(path: Path, ctx: ExtractContext) -> ExtractedDoc:
    from docling_core.types.doc import PictureItem, SectionHeaderItem, TableItem, TextItem

    from rag.core.ingest.figures import make_figure_block

    result = _get_converter(ctx).convert(long_path(path))
    doc = result.document
    blocks: list[Block] = []
    figure_index = 0
    for item, _depth in doc.iterate_items():
        page = item.prov[0].page_no if getattr(item, "prov", None) else None
        if isinstance(item, SectionHeaderItem):
            blocks.append(Block(kind="heading", text=item.text.strip(), level=int(getattr(item, "level", 1) or 1), page=page))
        elif isinstance(item, TableItem):
            df = item.export_to_dataframe(doc=doc)
            headers = [str(c) for c in df.columns]
            rows = [["" if v is None else str(v) for v in r] for r in df.itertuples(index=False)]
            caption = item.caption_text(doc) or None
            text = (f"**{caption}**\n\n" if caption else "") + markdown_table(headers, rows)
            blocks.append(Block(kind="table", text=text, page=page,
                                meta={"headers": headers, "rows": rows, "caption": caption}))
        elif isinstance(item, PictureItem):
            if (_picture_label(item) or "") in _SKIP_PICTURE_LABELS:
                continue
            img = item.get_image(doc)
            if img is None:
                continue
            figure_index += 1
            caption = item.caption_text(doc) or None
            blocks.append(make_figure_block(img, ctx, page=page, index=figure_index, caption=caption))
        elif isinstance(item, TextItem):
            label = str(getattr(item, "label", "")).lower()
            if "page_header" in label or "page_footer" in label:
                continue
            text = item.text.strip()
            if not text:
                continue
            if label.endswith("title"):
                blocks.append(Block(kind="heading", text=text, level=1, page=page))
            elif "list_item" in label:
                blocks.append(Block(kind="paragraph", text=f"- {text}", page=page))
            elif "code" in label:
                blocks.append(Block(kind="code", text=f"```\n{text}\n```", page=page))
            else:
                blocks.append(Block(kind="paragraph", text=text, page=page))
    n_pages = len(doc.pages) if getattr(doc, "pages", None) else None
    title = next((b.text for b in blocks if b.kind == "heading" and b.level == 1), None)
    return ExtractedDoc(file_type="pdf", title=title, blocks=blocks,
                        n_pages=n_pages, extractor="pdf-docling")
