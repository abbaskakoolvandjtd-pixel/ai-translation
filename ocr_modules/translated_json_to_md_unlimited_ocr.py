"""
Generate Markdown from the JSON detection output.

Usage:
    python json_to_md.py <output.json> [--pdf original.pdf] [--dpi 600] [--no-index] [--lang fa]

    --pdf       Original PDF (needed to crop images). If omitted, images are skipped.
    --dpi       DPI used during original extraction (for image cropping). Default: 600
    --no-index  Omit <!-- obj#N type --> comments
    --output    Output .md path (default: <json_stem>.md)
    --lang      Language code for translated content (e.g., 'fa' uses 'content_fa'). Default: fa

pip install PyMuPDF Pillow
"""
import os
import re
import sys
import json
import base64
import argparse
from pathlib import Path
from io import BytesIO
from collections import defaultdict
from typing import Optional, Union

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from PIL import Image
except ImportError:
    Image = None


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
Y_BAND_TOLERANCE = 20  # in 0-999 normalized space (~2% of page height)


# ──────────────────────────────────────────────
# Regex helpers (same as extraction script)
# ──────────────────────────────────────────────
REF_OPEN_RE     = re.compile(r'<\|ref\|>')
REF_CLOSE_RE    = re.compile(r'<\|/ref\|>')
ANY_SPECIAL_RE  = re.compile(r'<\|[^|]*\|>')
INLINE_MATH_RE  = re.compile(r'\\\(\s*(.*?)\s*\\\)', re.DOTALL)
DISPLAY_MATH_RE = re.compile(r'\\\[\s*(.*?)\s*\\\]', re.DOTALL)
CITATION_RE     = re.compile(r'^\[[\d,\s\u2013\u2014\-]+\]$')


# ──────────────────────────────────────────────
# HTML table → Markdown
# ──────────────────────────────────────────────
from html.parser import HTMLParser

class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] = []
        self._cell: list[str] = []
        self._in_cell = False

    def handle_starttag(self, tag, attrs):
        if tag == 'tr':
            self._row = []
        elif tag in ('td', 'th'):
            self._in_cell = True
            self._cell = []

    def handle_endtag(self, tag):
        if tag in ('td', 'th'):
            self._in_cell = False
            self._row.append(''.join(self._cell).strip())
        elif tag == 'tr' and self._row:
            self.rows.append(self._row)

    def handle_data(self, data):
        if self._in_cell:
            self._cell.append(data)


def html_table_to_md(html: str) -> str:
    p = _TableParser()
    p.feed(html)
    if not p.rows:
        return html
    ncols = max(len(r) for r in p.rows)
    for r in p.rows:
        while len(r) < ncols:
            r.append('')
    lines = [
        '| ' + ' | '.join(p.rows[0]) + ' |',
        '| ' + ' | '.join([':---:'] * ncols) + ' |',
    ]
    for r in p.rows[1:]:
        lines.append('| ' + ' | '.join(r) + ' |')
    return '\n'.join(lines)


# ──────────────────────────────────────────────
# Inline helpers
# ──────────────────────────────────────────────
def heading_level(text: str, y1: int | None) -> int:
    m = re.match(r'^(\d+(?:\.\d+)*)\.?\s', text)
    if m:
        return min(m.group(1).count('.') + 1, 6)
    if y1 is not None and y1 < 160:
        return 1
    return 2


def convert_inline(text: str) -> str:
    def _repl(m):
        inner = m.group(1).strip()
        if CITATION_RE.match(inner):
            return inner
        return f'${inner}$'
    return INLINE_MATH_RE.sub(_repl, text)


def clean_content(text: str) -> str:
    text = REF_OPEN_RE.sub('', text)
    text = REF_CLOSE_RE.sub('', text)
    text = ANY_SPECIAL_RE.sub('', text)
    return text


# ──────────────────────────────────────────────
# Reading-order sort
# ──────────────────────────────────────────────
def sort_objects_reading_order(objects: list[dict]) -> list[dict]:
    """
    Sort objects within a single page by visual reading order:
      primary  → vertical position (y1), banded for tolerance
      secondary → horizontal position (x1)
    Objects without bbox go to the end.
    """
    def sort_key(obj):
        bbox = obj.get("bbox")
        if bbox is None or len(bbox) != 4:
            return (99999, 99999)
        y_band = bbox[1] // Y_BAND_TOLERANCE
        return (y_band, bbox[0])

    return sorted(objects, key=sort_key)


# ──────────────────────────────────────────────
# Image cropping (optional, needs PDF + PyMuPDF + Pillow)
# ──────────────────────────────────────────────
def norm_to_pixel(bbox: tuple, img_w: int, img_h: int) -> tuple:
    x1, y1, x2, y2 = bbox
    px1 = max(0, int(x1 / 999 * img_w))
    py1 = max(0, int(y1 / 999 * img_h))
    px2 = min(img_w, int(x2 / 999 * img_w))
    py2 = min(img_h, int(y2 / 999 * img_h))
    return px1, py1, px2, py2


def crop_image_from_page(page: "fitz.Page", bbox: tuple, dpi: int) -> bytes | None:
    """Crop a region from a PDF page rendered at given DPI."""
    if fitz is None or Image is None:
        return None
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    pw, ph = pix.width, pix.height
    px1, py1, px2, py2 = norm_to_pixel(bbox, pw, ph)
    if px2 <= px1 or py2 <= py1:
        return None
    img = Image.frombytes("RGB", (pw, ph), pix.samples)
    cropped = img.crop((px1, py1, px2, py2))
    buf = BytesIO()
    cropped.save(buf, format="PNG")
    return buf.getvalue()


# ──────────────────────────────────────────────
# Core: JSON → Markdown
# ──────────────────────────────────────────────
def json_to_markdown(
    data: dict,
    pdf_doc: "fitz.Document | None" = None,
    dpi: int = 600,
    images_dir: Path | None = None,
    show_index: bool = True,
    lang: str = "fa",
) -> str:
    """
    Convert the detection JSON payload to Markdown.
    Mirrors the behaviour of the original CLI script.
    """
    parts: list[str] = []
    img_counter = 0
    images_dir = Path(str(images_dir))
    lang_key = f"content_{lang}"

    for pg_data in data.get("pages", []):
        pg_idx  = pg_data["page"]
        objects = pg_data.get("objects", [])

        # ── Sort into reading order ──
        # objects = sort_objects_reading_order(objects)

        # ── Page separator (optional, uncomment if desired) ──
        # parts.append(f"\n---\n<!-- PAGE {pg_idx} -->\n")

        for obj_idx, obj in enumerate(objects):
            etype   = obj.get("type", "text")
            bbox    = obj.get("bbox")

            # ── Skip objects with null bbox ──
            if bbox is None:
                continue

            # Get translated content if available, fallback to original content

            raw_content = obj.get(lang_key)
            if raw_content is None:
                raw_content = obj.get("content", "")
            content = clean_content(str(raw_content) if raw_content else "")

            idx_tag = f"<!-- obj#{obj_idx} {etype} -->\n" if show_index else ""

            # ── Skip non-content types ──
            if etype in ("header", "page_number", "footer"):
                continue

            # ── Title ──
            if etype == "title":
                lvl = heading_level(content, bbox[1] if bbox else None)
                parts.append(f"{idx_tag}{'#' * lvl} {content}")

            # ── Body text ──
            elif etype == "text":
                parts.append(f"{idx_tag}{convert_inline(content)}")


            # ── Equation ──
            elif etype == "equation":
                c = DISPLAY_MATH_RE.sub(r'$$\1$$', content)
                if not c.strip().startswith('$$'):
                    c = f"$$\n{c.strip()}\n$$"
                parts.append(f"{idx_tag}{c}")

            # ── Table ──
            elif etype == "table":
                if bbox and pdf_doc is not None and images_dir is not None:
                    img_counter += 1

                    page = pdf_doc[pg_idx] if pg_idx < len(pdf_doc) else None
                    img_bytes = crop_image_from_page(page, bbox, dpi) if page else None

                    if img_bytes:
                        # Save image
                        fname = f"page{pg_idx:03d}_img{img_counter:03d}.png"
                        fpath = images_dir / fname
                        fpath.write_bytes(img_bytes)

                        # Relative path from the Markdown file
                        rel = f"{images_dir.name}/{fname}"

                        # Calculate display width
                        # PDF points -> pixels assuming 96 DPI
                        w_pts = (bbox[2] - bbox[0]) / 999.0 * page.rect.width
                        w_px = max(1, int(w_pts * 96 / 72))

                        # Standard Markdown/Pandoc image syntax
                        # parts.append(
                        #     f'{idx_tag}![image_{img_counter}]({rel}){{width={w_px}px}}'
                        # )
                        parts.append(
                            f'{idx_tag}'
                            f'::: {{custom-style="Figure"}}\n\n'
                            f'![]({rel}){{width={w_px}px}}\n\n'
                            f':::'
                        )
                if '<table>' in content.lower():
                    content = html_table_to_md(content)
                parts.append(f"{idx_tag}{content}")

            # ── Image ──
            elif etype == "image":
                if bbox and pdf_doc is not None and images_dir is not None:
                    img_counter += 1

                    page = pdf_doc[pg_idx] if pg_idx < len(pdf_doc) else None
                    img_bytes = crop_image_from_page(page, bbox, dpi) if page else None

                    if img_bytes:
                        # Save image
                        fname = f"page{pg_idx:03d}_img{img_counter:03d}.png"
                        fpath = images_dir / fname
                        fpath.write_bytes(img_bytes)

                        # Relative path from the Markdown file
                        rel = f"{images_dir.name}/{fname}"

                        # Calculate display width
                        # PDF points -> pixels assuming 96 DPI
                        w_pts = (bbox[2] - bbox[0]) / 999.0 * page.rect.width
                        w_px = max(1, int(w_pts * 96 / 72))

                        # Standard Markdown/Pandoc image syntax
                        # parts.append(
                        #     f'{idx_tag}![image_{img_counter}]({rel}){{width={w_px}px}}'
                        # )
                        parts.append(
                            f'{idx_tag}'
                            f'::: {{custom-style="Figure"}}\n\n'
                            f'![]({rel}){{width={w_px}px}}\n\n'
                            f':::'
                        )

                    else:
                        parts.append(
                            f"{idx_tag}<!-- [image: crop failed] -->"
                        )
                else:
                    parts.append(
                        f"{idx_tag}<!-- [image: no source] -->"
                    )

            # ── Image caption ──
            elif etype == "image_caption":
                parts.append(f"{idx_tag}*{convert_inline(content)}*")

            # ── References ──
            elif etype == "ref_text":
                parts.append(
                    f'{idx_tag}'
                    f'::: {{ dir="ltr"}}\n\n'
                    f'{convert_inline(content)}\n\n'
                    f':::'
                )

            # ── Fallback ──
            else:
                if content:
                    parts.append(f"{idx_tag}{content}")

    md = '\n\n'.join(parts)
    md = re.sub(r'\n{4,}', '\n\n\n', md)
    if lang == "fa":
        md = """---
dir: rtl
lang: fa
---
"""+md
    return md.strip()


# ──────────────────────────────────────────────
# Public helper – callable from other code
# ──────────────────────────────────────────────
def run_json_to_md(
    json_path: Union[str, Path],
    pdf_path: Optional[Union[str, Path]] = None,
    dpi: int = 600,
    no_index: bool = False,
    output_path: Optional[Union[str, Path]] = None,
    lang: str = "fa",
) -> Path:
    """
    Perform the full JSON‑to‑Markdown conversion.

    Returns the Path to the generated Markdown file.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON not found: {json_path}")

    # Load JSON
    print(f"📂 Loading {json_path} ...")
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    total_pages   = data.get("total_pages", 0)
    total_objects = data.get("total_objects", 0)
    print(f"   {total_pages} pages, {total_objects} objects")

    # Open PDF (optional)
    pdf_doc = None
    if pdf_path:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            print(f"⚠️  PDF not found: {pdf_path} — images will be skipped")
        elif fitz is None:
            print("⚠️  PyMuPDF not installed — images will be skipped")
        else:
            pdf_doc = fitz.open(str(pdf_path))
            print(f"📄 Opened {pdf_path} ({len(pdf_doc)} pages)")

    # Prepare images directory
    stem = json_path.stem
    main_dir_images = os.path.dirname(
        json_path)
    images_dir_path = os.path.join(main_dir_images,f"{stem}_images")
    if pdf_doc is not None:
        if not os.path.exists(images_dir_path):
            os.mkdir(images_dir_path)

    images_dir_path = Path(str(images_dir_path))
    # Generate Markdown
    print(f"📝 Generating Markdown (Language: {lang}) ...")
    markdown = json_to_markdown(
        data,
        pdf_doc=pdf_doc,
        dpi=dpi,
        images_dir=images_dir_path if pdf_doc else None,
        show_index=not no_index,
        lang=lang,
    )

    # Write output
    out_path = Path(output_path) if output_path else Path(f"{stem}.md")
    out_path.write_text(markdown, encoding="utf-8")

    n_imgs = len(list(images_dir_path.glob("*.png"))) if images_dir_path.exists() else 0
    print(f"\n💾 Markdown → {out_path}  ({len(markdown)} chars, {n_imgs} images)")

    if pdf_doc:
        pdf_doc.close()

    print("✅ Done!")
    return out_path


# ──────────────────────────────────────────────
# CLI entry point (preserves original behaviour)
# ──────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Markdown from JSON detection output")
    parser.add_argument("json", help="JSON file from the extraction pipeline")
    parser.add_argument("--pdf", default=None,
                        help="Original PDF (for cropping images)")
    parser.add_argument("--dpi", type=int, default=600,
                        help="DPI for image cropping (default: 600)")
    parser.add_argument("--no-index", action="store_true",
                        help="Omit <!-- obj#N --> comments")
    parser.add_argument("--output", "-o", default=None,
                        help="Output .md path (default: <json_stem>.md)")
    parser.add_argument("--lang", default="fa",
                        help="Language code for translated content (e.g. 'fa' for content_fa). Default: fa")
    args = parser.parse_args()

    # Delegate to the reusable helper
    run_json_to_md(
        json_path=args.json,
        pdf_path=args.pdf,
        dpi=args.dpi,
        no_index=args.no_index,
        output_path=args.output,
        lang=args.lang,
    )


if __name__ == "__main__":
    # main()

    run_json_to_md(
            json_path=r"E:\Kakoolvand\pycharm_projects\markitdown_project\unlimited_ocr_output_files\World_Development_Report_2026_The_Promise_of_Artificial_Intelligence_persian.json",
            pdf_path=r"E:\Kakoolvand\pycharm_projects\markitdown_project\World_Development_Report_2026_The_Promise_of_Artificial_Intelligence.pdf",
            dpi= 600,
            output_path= r"E:\Kakoolvand\pycharm_projects\markitdown_project\unlimited_ocr_output_files\World_Development_Report_2026_The_Promise_of_Artificial_Intelligence_persian.md",
            lang="fa",
    )