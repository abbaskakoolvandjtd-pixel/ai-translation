"""
Single-Page PDF → Markdown + JSON via vLLM + Unlimited-OCR (MAXIMUM PRECISION)

Optimizations for Highest Accuracy & Memory Management:
  1. BATCH_SIZE = 1: Forces single-image "gundam (crop) mode" for every page.
  2. MAX_TOKENS = 32768: Prevents any possibility of truncation on dense pages.
  3. Hardcoded window_size=128: Optimal for single-page context tracking.
  4. temperature=0.0 & skip_special_tokens=False: Enforced for deterministic,
     repetition-free extraction with intact grounding tokens.
  5. STREAMING MEMORY: Processes 1 page at a time to prevent RAM exhaustion on large PDFs.

pip install openai PyMuPDF Pillow
"""

import os
import re
import sys
import json
import time
import base64
import gc  # <-- Garbage collection to free memory between pages
import logging
import fitz  # PyMuPDF
from pathlib import Path
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI
from html.parser import HTMLParser

from core.db import update_job_status
from core.utils import TranslationConfig

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Config (Optimized for Maximum Precision)
# ──────────────────────────────────────────────
VLLM_URL = getattr(TranslationConfig, "VLLM_URL", "http://10.19.24.49:5090/v1")
API_KEY = getattr(TranslationConfig, "VLLM_API_KEY", None)
if not API_KEY:
    raise ValueError("VLLM_API_KEY must be set in environment variables or .env file")
    
MODEL_NAME = getattr(TranslationConfig, "OCR_MODEL_NAME", "unlimited-ocr")
DPI = int(getattr(TranslationConfig, "OCR_DPI", 600))
MAX_TOKENS = int(getattr(TranslationConfig, "OCR_MAX_TOKENS", 10000))
BATCH_SIZE = int(getattr(TranslationConfig, "OCR_BATCH_SIZE", 1))
DEBUG = getattr(TranslationConfig, "OCR_DEBUG", False)

client = OpenAI(api_key=API_KEY, base_url=VLLM_URL, timeout=3600)


# ══════════════════════════════════════════════
# 2. Coordinate mapping
# ══════════════════════════════════════════════
def norm_to_pixel(bbox: tuple, img_w: int, img_h: int) -> tuple:
    """Map normalised [0-999] bbox → pixel coords on the original image."""
    x1, y1, x2, y2 = bbox
    px1 = int(x1 / 999 * img_w)
    py1 = int(y1 / 999 * img_h)
    px2 = int(x2 / 999 * img_w)
    py2 = int(y2 / 999 * img_h)
    # clamp
    px1, py1 = max(0, px1), max(0, py1)
    px2, py2 = min(img_w, px2), min(img_h, py2)
    return px1, py1, px2, py2


def crop_bbox(pix: fitz.Pixmap, bbox: tuple) -> bytes:
    """Crop a region from a page pixmap using normalised coords."""
    pw, ph = pix.width, pix.height
    px1, py1, px2, py2 = norm_to_pixel(bbox, pw, ph)

    if px2 <= px1 or py2 <= py1:
        return b""

    img = Image.frombytes("RGB", (pw, ph), pix.samples)
    cropped = img.crop((px1, py1, px2, py2))
    buf = BytesIO()
    cropped.save(buf, format="PNG")
    return buf.getvalue()


# ══════════════════════════════════════════════
# 3. Debug: draw all bboxes on a page
# ══════════════════════════════════════════════
COLORS = {
    "title":        (255,  0,  0),
    "text":         (  0, 128,  0),
    "image":        (  0,  0, 255),
    "table":        (255, 165,  0),
    "equation":     (128,  0, 128),
    "image_caption":(  0, 200, 200),
    "ref_text":     (128, 128, 128),
    "header":       (200, 200,  0),
    "page_number":  (200, 100, 100),
}


def save_debug_page(img_path: Path, blocks: list[dict],
                    page_idx: int, debug_dir: Path):
    """Save the page image with all bounding boxes drawn."""
    debug_dir.mkdir(parents=True, exist_ok=True)
    # We load from disk here since we freed the pixmap from RAM
    img = Image.open(img_path).convert("RGB")
    pw, ph = img.size
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except (IOError, OSError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
        except (IOError, OSError):
            font = ImageFont.load_default()

    for blk in blocks:
        if blk.get("bbox") is None:
            continue
        etype = blk["type"]
        color = COLORS.get(etype, (100, 100, 100))
        px1, py1, px2, py2 = norm_to_pixel(blk["bbox"], pw, ph)
        draw.rectangle([px1, py1, px2, py2], outline=color, width=4)
        label = f"{etype} {blk['bbox']}"
        draw.text((px1 + 4, max(0, py1 - 32)), label, fill=color, font=font)

    out = debug_dir / f"debug_page_{page_idx:03d}.png"
    img.save(str(out))
    return out


# ══════════════════════════════════════════════
# 4. Regex patterns
# ══════════════════════════════════════════════
DET_RE = re.compile(
    r'<\|det\|>\s*(\w+)\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\s*<\|/det\|>(.*)',
    re.DOTALL,
)
PAGE_RE        = re.compile(r'<PAGE>')
REF_OPEN_RE    = re.compile(r'<\|ref\|>')
REF_CLOSE_RE   = re.compile(r'<\|/ref\|>')
ANY_SPECIAL_RE = re.compile(r'<\|[^|]*\|>')
INLINE_MATH_RE = re.compile(r'\\\(\s*(.*?)\s*\\\)', re.DOTALL)
DISPLAY_MATH_RE= re.compile(r'\\\[\s*(.*?)\s*\\\]', re.DOTALL)
CITATION_RE    = re.compile(r'^\[[\d,\s\u2013\u2014\-]+\]$')


# ══════════════════════════════════════════════
# 5. HTML table → Markdown
# ══════════════════════════════════════════════
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


def _html_table_to_md(html: str) -> str:
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
        '| ' + ' | '.join(['---'] * ncols) + ' |',
    ]
    for r in p.rows[1:]:
        lines.append('| ' + ' | '.join(r) + ' |')
    return '\n'.join(lines)


# ══════════════════════════════════════════════
# 6. Helpers
# ══════════════════════════════════════════════
def _heading_level(text: str, y1: int) -> int:
    m = re.match(r'^(\d+(?:\.\d+)*)\.?\s', text)
    if m:
        return min(m.group(1).count('.') + 1, 6)
    return 1 if y1 < 160 else 2


def _convert_inline(text: str) -> str:
    def _repl(m):
        inner = m.group(1).strip()
        if CITATION_RE.match(inner):
            return inner
        return f'${inner}$'
    return INLINE_MATH_RE.sub(_repl, text)


# ══════════════════════════════════════════════
# 7. Parse raw OCR → structured blocks
# ══════════════════════════════════════════════
def parse_raw_ocr(raw: str, page_offset: int = 0) -> list[dict]:
    lines = raw.split('\n')
    blocks: list[dict] = []
    cur_lines: list[str] = []
    cur_type: str | None = None
    cur_bbox: tuple | None = None
    current_page = 0

    def flush():
        nonlocal cur_lines, cur_type, cur_bbox
        if cur_lines:
            txt = '\n'.join(cur_lines).strip()
            if txt:
                blocks.append({
                    "page": current_page + page_offset,
                    "type": cur_type,
                    "bbox": cur_bbox,
                    "content": txt,
                })
        cur_lines, cur_type, cur_bbox = [], None, None

    for line in lines:
        s = line.rstrip()

        if PAGE_RE.search(s):
            flush()
            current_page += 1
            s = PAGE_RE.sub('', s).strip()
            if not s:
                continue

        m = DET_RE.match(s)
        if m:
            flush()
            etype = m.group(1)
            bbox = (int(m.group(2)), int(m.group(3)),
                    int(m.group(4)), int(m.group(5)))
            rest = m.group(6).strip()

            if etype == 'image' or etype == 'chart':
                blocks.append({
                    "page": current_page + page_offset,
                    "type": "image",
                    "bbox": bbox,
                    "content": "",
                })
                continue
            if etype == 'list':
                continue

            cur_type, cur_bbox = etype, bbox
            cur_lines = [rest] if rest else []
            continue

        if cur_type is not None:
            cur_lines.append(s)
        elif s:
            # Ignoring orphan text lines – they are usually artifacts.
            pass

    flush()
    return blocks


# ══════════════════════════════════════════════
# 8. Blocks → Markdown (with image embedding)
# ══════════════════════════════════════════════
def blocks_to_markdown(blocks: list[dict], images_dir: Path) -> str:
    images_dir.mkdir(parents=True, exist_ok=True)
    parts: list[str] = []

    for blk in blocks:
        etype    = blk["type"]
        bbox     = blk["bbox"]
        content  = blk["content"]
        page_idx = blk["page"]

        content = REF_OPEN_RE.sub('', content)
        content = REF_CLOSE_RE.sub('', content)
        content = ANY_SPECIAL_RE.sub('', content)

        if etype == 'header':
            continue
        elif etype == 'title':
            lvl = _heading_level(content, bbox[1] if bbox else 200)
            parts.append(f"{'#' * lvl} {content}")
        elif etype == 'text':
            parts.append(_convert_inline(content))
        elif etype == 'equation':
            c = DISPLAY_MATH_RE.sub(r'$$\1$$', content)
            if not c.strip().startswith('$$'):
                c = f"$$\n{c.strip()}\n$$"
            parts.append(c)
        elif etype == 'table':
            if '<table>' in content.lower():
                content = _html_table_to_md(content)
            parts.append(content)
        elif etype == 'image':
            # Retrieve the already-cropped and saved image file
            fname = blk.get("image_file")
            if fname:
                rel = f"{images_dir.name}/{fname}"
                parts.append(f"![image_{blk.get('img_id', '')}]({rel})")
            else:
                parts.append('<!-- [image: crop failed or no bbox] -->')
        elif etype == 'image_caption':
            parts.append(f"*{_convert_inline(content)}*")
        elif etype == 'page_number':
            continue
        elif etype == 'ref_text':
            parts.append(content)
        else:
            if content:
                parts.append(content)

    md = '\n\n'.join(parts)
    return re.sub(r'\n{4,}', '\n\n\n', md).strip()


# ══════════════════════════════════════════════
# 9. Blocks → JSON
# ══════════════════════════════════════════════
def blocks_to_json(blocks: list[dict]) -> dict:
    pages_dict: dict[int, list] = {}
    for blk in blocks:
        pg = blk["page"]
        obj = {
            "type":    blk["type"],
            "bbox":    list(blk["bbox"]) if blk["bbox"] else None,
            "content": blk["content"],
        }
        if "image_file" in blk:
            obj["image_file"] = blk["image_file"]
        pages_dict.setdefault(pg, []).append(obj)

    pages_list = []
    for pg_idx in sorted(pages_dict.keys()):
        pages_list.append({
            "page":        pg_idx,
            "num_objects": len(pages_dict[pg_idx]),
            "objects":     pages_dict[pg_idx],
        })

    return {
        "total_pages":   len(pages_list),
        "total_objects": len(blocks),
        "pages":         pages_list,
    }

# ══════════════════════════════════════════════
# 10. Synchronous single page (Non-streaming)
# ══════════════════════════════════════════════
def ocr_page(image_b64: str, prompt: str, window_size: int) -> str:
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": content}],
        max_tokens=MAX_TOKENS,
        temperature=0.0,
        extra_body={
            "skip_special_tokens": False,
            "vllm_xargs": {
                "ngram_size": 35,
                "window_size": window_size,
            },
        },
    )

    raw_text = response.choices[0].message.content
    logger.info(raw_text)
    return raw_text if raw_text else ""


# ══════════════════════════════════════════════
# 11. Core processing function (callable from other modules)
# ══════════════════════════════════════════════
def process_pdf(pdf_path: str,job_id=None) -> dict:
    """
    Executes the full OCR → Markdown/JSON pipeline for *pdf_path*.
    Refactored to process one page at a time to prevent RAM exhaustion.
    """
    logger.info(f"📄 Converting {pdf_path} → images (dpi={DPI}) ...")
    doc = fitz.open(pdf_path)
    n_pages = len(doc)

    logger.info(f"   {n_pages} page(s) → Processing 1 page per API call for MAXIMUM PRECISION\n")

    images_dir = Path(IMAGES_DIR)
    images_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = Path(pdf_path).with_suffix("") / "_temp_pages"
    if DEBUG:
        temp_dir.mkdir(parents=True, exist_ok=True)

    all_blocks: list[dict] = []
    total_time = 0.0
    global_img_counter = 0

    mat = fitz.Matrix(DPI / 72, DPI / 72)

    for i in range(n_pages):
        page_idx = i
        page = doc[page_idx]
        pix = page.get_pixmap(matrix=mat)

        # Get base64 directly without saving a massive byte array to a list
        image_b64 = base64.b64encode(pix.tobytes("png")).decode()

        if DEBUG:
            pix.save(temp_dir / f"page_{page_idx:04d}.png")

        logger.info(f"\n{'═' * 60}")
        logger.info(f"🚀 Processing Page {page_idx + 1} of {n_pages} (Global Index: {page_idx})")
        logger.info(f"{'═' * 60}\n")
        percentage=int(100*((page_idx + 1)/n_pages))
        if job_id:
            update_job_status(job_id, "OCR_PROCESSING", status_detail=f"""در حال OCR فایل  /   صفحه {page_idx + 1} از {n_pages} صفحه ({percentage}٪)""")

        prompt = "<image>document parsing."
        ws = 128
        t0 = time.time()

        try:
            raw = ocr_page(image_b64, prompt, ws)
        except Exception as ocr_err:
            # 🔥 Update DB with the exact error and page number before crashing
            error_msg = f"OCR LLM failed on page {page_idx + 1}/{n_pages}. Error: {str(ocr_err)}"
            logger.error(f"{error_msg}")
            if job_id:
                # This will immediately set the DB state to FAILED via the executor's catch block,
                # but we update the detail here for better logging in the DB.
                update_job_status(job_id, "FAILED", error_message=error_msg)
            raise  # Re-raise to crash the pipeline and trigger pipeline_executor's except block

        elapsed = time.time() - t0
        total_time += elapsed
        logger.info(f"\n⏱  {elapsed:.1f}s  |  {len(raw)} chars raw")

        page_blocks = parse_raw_ocr(raw, page_offset=page_idx)

        # Crop images and save to final directory immediately
        for blk in page_blocks:
            if blk["type"] == 'image' and blk["bbox"]:
                global_img_counter += 1
                img_bytes = crop_bbox(pix, blk["bbox"])
                if img_bytes:
                    fname = f"page{page_idx:03d}_img{global_img_counter:03d}.png"
                    fpath = images_dir / fname
                    fpath.write_bytes(img_bytes)
                    blk["image_file"] = fname
                    blk["img_id"] = global_img_counter
                else:
                    blk["image_file"] = None
            elif blk["type"] == 'image':
                blk["image_file"] = None

        all_blocks.extend(page_blocks)
        logger.info(f"   → {len(page_blocks)} blocks parsed")

        # Critical Step: Free memory from pixmap before moving to the next page
        del pix
        gc.collect()

    doc.close()

    logger.info(f"\n{'─' * 60}")
    logger.info(f"📊 Total: {len(all_blocks)} blocks, "
          f"{max((b['page'] for b in all_blocks), default=-1) + 1} pages, "
          f"{total_time:.1f}s total processing time")

    debug_dir = None
    if DEBUG:
        debug_dir = Path(DEBUG_DIR)
        logger.info(f"\n🔍 Saving debug annotations → {debug_dir}/")
        pages_with_blocks: dict[int, list] = {}
        for blk in all_blocks:
            pages_with_blocks.setdefault(blk["page"], []).append(blk)
        for pg_idx, blks in sorted(pages_with_blocks.items()):
            debug_img_path = temp_dir / f"page_{pg_idx:04d}.png"
            if debug_img_path.exists():
                out = save_debug_page(debug_img_path, blks, pg_idx, debug_dir)
                logger.info(f"page {pg_idx}: {len(blks)} objects → {out.name}")

    markdown = blocks_to_markdown(all_blocks, images_dir)
    json_out = blocks_to_json(all_blocks)

    return {
        "markdown": markdown,
        "json": json_out,
        "images_dir": images_dir,
        "debug_dir": debug_dir,
        "total_time": total_time,
        "total_blocks": len(all_blocks),
    }


# ══════════════════════════════════════════════
# 12. CLI entry point
# ══════════════════════════════════════════════
def main(pdf_path=None):
    global PDF_PATH, OUTPUT_MD, OUTPUT_JSON, IMAGES_DIR, DEBUG_DIR
    if pdf_path:
        PDF_PATH = pdf_path
        OUTPUT_MD = PDF_PATH.rsplit(".", 1)[0] + ".md"
        OUTPUT_JSON = PDF_PATH.rsplit(".", 1)[0] + ".json"
        IMAGES_DIR = PDF_PATH.rsplit(".", 1)[0] + "_images"
        DEBUG_DIR = PDF_PATH.rsplit(".", 1)[0] + "_debug"
    pdf = PDF_PATH

    if len(sys.argv) > 1:
        pdf = sys.argv[1]

    result = process_pdf(pdf)

    # Persist Markdown
    Path(OUTPUT_MD).write_text(result["markdown"], encoding="utf-8")
    n_imgs = len(list(result["images_dir"].glob("*.png")))
    logger.info(f"\n💾 Markdown → {OUTPUT_MD}  ({len(result['markdown'])} chars, {n_imgs} images)")

    # Persist JSON
    Path(OUTPUT_JSON).write_text(
        json.dumps(result["json"], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(f"💾 JSON     → {OUTPUT_JSON}  "
          f"({result['json']['total_objects']} objects, "
          f"{result['json']['total_pages']} pages)")

    logger.info("\n✅ Done!")


if __name__ == "__main__":
    tic = time.time()
    main()
    toc = time.time()
    logger.info(f"Total elapsed time: {int(toc-tic)}s")

