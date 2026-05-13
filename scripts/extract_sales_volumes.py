"""
Extract sales volumes (Pkgs) per grade from the monthly KTDA tea-auction PDFs
in `salesreport/` and merge them, together with monthly USD/KES exchange
rates, into the master CSV consumed by the forecasting engine.

Output: analytics/data/tea_auction_data.csv (overwritten with extracted data)
Cache:  analytics/data/.fx_cache.json (so we don't hammer the FX API)

Run from the project root with the venv active:

    python scripts/extract_sales_volumes.py
    python scripts/extract_sales_volumes.py --dry-run   # preview only
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import urllib.request
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import pdfplumber

# OCR engine — lazy-initialised only if we hit an image-only PDF.
_OCR_ENGINE = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("extract")

ROOT       = Path(__file__).resolve().parent.parent
PDF_DIR    = ROOT / "salesreport"
CSV_PATH   = ROOT / "analytics" / "data" / "tea_auction_data.csv"
FX_CACHE   = ROOT / "analytics" / "data" / ".fx_cache.json"
OCR_CACHE  = ROOT / "analytics" / "data" / ".ocr_cache.json"

# Header tokens as they appear in the PDFs (with TOTALS at the end).
# Order matters: must match the column ordering in the PDF totals row.
HEADER_GRADES = ["BP1", "PF1", "PD", "DUST1", "FNGS1/2", "DUST/2", "BMF"]

# Map PDF header tokens → CSV column suffixes used by the model.
GRADE_TO_CSV = {
    "BP1":     "BP1",
    "PF1":     "PF1",
    "DUST1":   "DUST1",
    "FNGS1/2": "FNGS_1_2",
    "DUST/2":  "DUST_1_2",
}
# We intentionally drop PD and BMF: they are niche grades not in the CSV.

# Calendar months → 1-based index for parsing filenames like "May 2024.pdf".
MONTH_LOOKUP = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "september": 9, "sept": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


# ===========================================================================
# PDF parsing
# ===========================================================================
NUM_RE = re.compile(r"^-?[\d,]+(?:\.\d+)?$")
# Tolerant of OCR concatenation: "Auction Nos 2025/21 to 2025/21" can come
# back as "AuctionNoS2026/13to2026/13" after OCR, so allow optional
# whitespace and a permissive "to" boundary.
AUCTION_HEADER_RE = re.compile(
    r"Auction\s*Nos?\.?\s*(\d{4})\s*/\s*(\d{1,3})\s*(?:to|\-|\u2013)\s*(\d{4})\s*/\s*(\d{1,3})",
    re.I,
)
# Daily-sale image PDFs use "Auction No : 11 Sale Date : 13/03/2026".
# After OCR the spacing/punctuation can vary, so accept any non-word
# separators between "No" and the digits.
DAILY_AUCTION_RE  = re.compile(r"Auction\s*No\W*(\d{1,3})\b", re.I)
DAILY_FILENAME_RE = re.compile(r"sale\s+for\s+(\d{1,2})\.(\d{1,2})\.(\d{4})", re.I)


def _to_number(token: str) -> float | None:
    if not NUM_RE.match(token):
        return None
    try:
        return float(token.replace(",", ""))
    except ValueError:
        return None


def filename_to_period(name: str) -> date | None:
    """
    Resolve PDF filenames to a sale date.

    Monthly filenames ("Jan 2025.pdf", "May 2023.pdf", …) → first of the month.
    Daily filenames ("sale for 13.03.2026.pdf")            → actual sale date.
    """
    stem = Path(name).stem.lower().strip()

    m = DAILY_FILENAME_RE.search(stem)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return date(y, mo, d)
        except ValueError:
            return None

    parts = stem.replace("-", " ").split()
    month = None
    year = None
    for p in parts:
        if p in MONTH_LOOKUP and month is None:
            month = MONTH_LOOKUP[p]
        elif p.isdigit() and len(p) == 4 and year is None:
            year = int(p)
    if month is None or year is None:
        return None
    return date(year, month, 1)


def parse_totals_row(numbers: list[float]) -> dict[str, float] | None:
    """
    The PDF totals row has the structure::

        [Pkgs, Avg] × N grades  ··  TotalPkgs  TotalKgs.NN  TotalAvg

    TotalKgs is always written with a trailing .NN (e.g. ``4,583,566.00``),
    which lets us peg the boundary between grade pairs and overall totals.

    Returns a dict with the per-grade Pkgs (sales volume), or None on failure.
    """
    if len(numbers) < 5:
        return None

    # Walk backwards looking for the TotalKgs token (has fractional part > 100).
    # Heuristic: TotalKgs > 1000 and has a non-trivial fractional component
    # in the original string ⇒ we can't tell that from the float, so instead
    # use position: it's the second-from-last number.
    # numbers[-3] = TotalPkgs, numbers[-2] = TotalKgs, numbers[-1] = TotalAvg
    if len(numbers) < 5:
        return None

    pair_count = (len(numbers) - 3) // 2
    if pair_count < 1 or pair_count > len(HEADER_GRADES):
        return None

    result: dict[str, float] = {}
    # Assume left-to-right header order. If pair_count < 7, we drop trailing
    # grades (BMF first, then DUST/2, etc.). In practice BMF is the only one
    # that's ever absent in the totals row, so this is safe.
    for i in range(pair_count):
        grade = HEADER_GRADES[i]
        pkgs = numbers[2 * i]
        avg  = numbers[2 * i + 1]
        if grade in GRADE_TO_CSV:
            csv_key = GRADE_TO_CSV[grade]
            result[f"{csv_key}_pkgs"] = pkgs
            result[f"{csv_key}_avg"]  = avg

    result["total_pkgs"]   = numbers[-3]
    result["total_kgs"]    = numbers[-2]
    result["total_avg"]    = numbers[-1]
    return result


def _build_row(period: date, auction_no: str, parsed: dict, source_pdf: str) -> dict:
    """Assemble the final dict for one CSV row."""
    return {
        "date":       pd.Timestamp(period),
        "auction_no": auction_no or "",
        # Per-grade price (Avg from the totals row)
        **{g: parsed.get(f"{g}_avg") for g in GRADE_TO_CSV.values()},
        # Per-grade sales volume (Pkgs)
        **{f"{g}_pkgs": parsed.get(f"{g}_pkgs") for g in GRADE_TO_CSV.values()},
        # Aggregate volumes
        "total_pkgs": parsed.get("total_pkgs"),
        "total_kgs":  parsed.get("total_kgs"),
        "total_avg":  parsed.get("total_avg"),
        "source_pdf": source_pdf,
    }


def _extract_via_text(path: Path) -> tuple[str | None, list[float] | None]:
    """
    Pull the auction header and the bottom totals row out of a PDF using
    plain text extraction. Returns ``(auction_no, totals_numbers)`` where each
    component is None when nothing could be parsed.
    """
    try:
        pdf = pdfplumber.open(str(path))
    except Exception as e:
        log.error("Failed to open %s: %s", path.name, e)
        return None, None

    auction_no: str | None = None
    totals_numbers: list[float] | None = None
    try:
        for page in pdf.pages:
            text = page.extract_text() or ""
            for line in text.splitlines():
                if auction_no is None:
                    m = AUCTION_HEADER_RE.search(line)
                    if m:
                        auction_no = f"{m.group(1)}/{m.group(2)}"
                tokens = line.split()
                nums = [_to_number(t) for t in tokens]
                if nums and all(n is not None for n in nums) and len(nums) >= 7:
                    totals_numbers = [n for n in nums if n is not None]  # type: ignore[misc]
    finally:
        pdf.close()

    return auction_no, totals_numbers


def _get_ocr_engine():
    """Lazy-load the RapidOCR engine — heavy import, only needed for image PDFs."""
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "OCR fallback requires `rapidocr-onnxruntime` and `pypdfium2`. "
                "Install with: pip install rapidocr-onnxruntime pypdfium2"
            ) from e
        _OCR_ENGINE = RapidOCR()
    return _OCR_ENGINE


def _ocr_cache_load() -> dict:
    if OCR_CACHE.exists():
        try:
            return json.loads(OCR_CACHE.read_text())
        except Exception:
            return {}
    return {}


def _ocr_cache_save(cache: dict) -> None:
    OCR_CACHE.parent.mkdir(parents=True, exist_ok=True)
    OCR_CACHE.write_text(json.dumps(cache, indent=2, sort_keys=True))


def _ocr_pdf_pages(path: Path, scale: float = 300 / 72) -> list[list[tuple[float, float, str]]]:
    """
    OCR every page of an image-based PDF. Returns one list of
    ``(y_top, x_left, text)`` tokens per page, sorted top-to-bottom / left-to-right.

    Results are cached on disk keyed by ``(filename, mtime, size)`` so re-runs
    skip the expensive OCR step for unchanged PDFs.
    """
    cache = _ocr_pdf_pages._cache  # type: ignore[attr-defined]
    stat = path.stat()
    cache_key = f"{path.name}|{int(stat.st_mtime)}|{stat.st_size}"
    if cache_key in cache:
        return [[tuple(tok) for tok in page] for page in cache[cache_key]]

    try:
        import pypdfium2 as pdfium  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "OCR fallback requires `pypdfium2`. Install with: pip install pypdfium2"
        ) from e

    ocr = _get_ocr_engine()
    pdf = pdfium.PdfDocument(str(path))
    pages: list[list[tuple[float, float, str]]] = []
    try:
        for page_idx in range(len(pdf)):
            page = pdf[page_idx]
            pil = page.render(scale=scale).to_pil()
            try:
                result, _ = ocr(pil)
            except Exception as e:
                log.warning("OCR failed on %s page %d: %s", path.name, page_idx + 1, e)
                pages.append([])
                continue
            tokens: list[tuple[float, float, str]] = []
            if result:
                for entry in result:
                    box, text, _score = entry
                    ys = [pt[1] for pt in box]
                    xs = [pt[0] for pt in box]
                    tokens.append((min(ys), min(xs), text.strip()))
            tokens.sort()
            pages.append(tokens)
    finally:
        pdf.close()

    cache[cache_key] = [[list(tok) for tok in page] for page in pages]
    _ocr_cache_save(cache)
    return pages


# Initialise the in-memory OCR-page cache at import time.
_ocr_pdf_pages._cache = _ocr_cache_load()  # type: ignore[attr-defined]


def _extract_via_ocr(path: Path, period: date) -> tuple[str | None, list[float] | None]:
    """
    OCR-based fallback for image-only daily-sale PDFs.

    Strategy:
    1. OCR every page.
    2. Search the combined text for the auction header. Monthly-style header
       (``Auction Nos YYYY/NN to YYYY/NN``) wins if present; otherwise fall back
       to the daily-style header (``Auction No : NN``) and pair it with the
       sale-date year from the filename to build ``YYYY/NN``.
    3. On the last page, locate the y-band of the totals row by finding a
       token shaped like a 6+ digit total-kgs value (e.g. ``4,355,170.00``).
    4. Sort the tokens in that band left-to-right and return their numeric
       values to be parsed by ``parse_totals_row``.
    """
    log.info("OCR-decoding %s …", path.name)
    pages = _ocr_pdf_pages(path)
    if not any(pages):
        log.warning("OCR returned no text for %s", path.name)
        return None, None

    full_text = "\n".join(tok[2] for page in pages for tok in page)

    auction_no: str | None = None
    m = AUCTION_HEADER_RE.search(full_text)
    if m:
        auction_no = f"{m.group(1)}/{m.group(2)}"
    else:
        m2 = DAILY_AUCTION_RE.search(full_text)
        if m2:
            auction_no = f"{period.year}/{int(m2.group(1)):02d}"

    last_page = pages[-1]
    totals_y: float | None = None
    # Total kgs is always > 100,000 and ends with ".00".
    for y, _x, text in last_page:
        clean = text.replace(",", "").replace(" ", "")
        if re.fullmatch(r"\d+\.00", clean):
            try:
                val = float(clean)
            except ValueError:
                continue
            if val >= 100_000:
                totals_y = y
                break

    if totals_y is None:
        log.warning("Could not locate totals row via OCR in %s", path.name)
        return auction_no, None

    band = [t for t in last_page if abs(t[0] - totals_y) <= 20]
    band.sort(key=lambda t: t[1])
    nums = [n for n in (_to_number(t[2]) for t in band) if n is not None]
    if len(nums) < 7:
        log.warning("OCR totals row in %s has too few numbers: %s",
                    path.name, nums)
        return auction_no, None
    return auction_no, nums


def extract_pdf(path: Path) -> dict | None:
    """
    Extract one CSV row from a PDF — works for both monthly text PDFs and
    daily image-only PDFs (transparent OCR fallback).
    """
    period = filename_to_period(path.name)
    if period is None:
        log.warning("Skipping %s — could not parse period from filename.", path.name)
        return None

    auction_no, totals_numbers = _extract_via_text(path)

    if totals_numbers is None:
        # Image-only PDF — fall back to OCR.
        try:
            auction_no_ocr, totals_numbers = _extract_via_ocr(path, period)
        except RuntimeError as e:
            log.error("%s — %s", path.name, e)
            return None
        if auction_no_ocr and not auction_no:
            auction_no = auction_no_ocr

    if totals_numbers is None:
        log.warning("Could not locate totals row in %s", path.name)
        return None

    parsed = parse_totals_row(totals_numbers)
    if parsed is None:
        log.warning("Failed to parse totals row in %s: %s",
                    path.name, totals_numbers)
        return None

    return _build_row(period, auction_no or "", parsed, path.name)


# Backwards-compatible alias for any external caller.
extract_monthly_pdf = extract_pdf


# ===========================================================================
# FX rates (USD → KES, monthly)
# ===========================================================================
def _load_fx_cache() -> dict[str, float]:
    if FX_CACHE.exists():
        try:
            return json.loads(FX_CACHE.read_text())
        except Exception:
            return {}
    return {}


def _save_fx_cache(cache: dict[str, float]) -> None:
    FX_CACHE.write_text(json.dumps(cache, indent=2, sort_keys=True))


def fetch_usd_kes(d: date, cache: dict[str, float]) -> float | None:
    """
    Use exchangerate.host /historical endpoint (no key required) to fetch the
    1-USD→KES rate for the given date. Cached locally so we never re-fetch.

    If the historical endpoint is unavailable, the function returns None.
    """
    key = d.strftime("%Y-%m-%d")
    if key in cache:
        return cache[key]

    url = f"https://api.exchangerate.host/historical?date={key}&base=USD&symbols=KES"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            payload = json.loads(resp.read().decode())
    except Exception as e:
        log.warning("FX fetch failed for %s: %s", key, e)
        return None

    rate: float | None = None
    if payload.get("success") and isinstance(payload.get("quotes"), dict):
        rate = payload["quotes"].get("USDKES")
    elif isinstance(payload.get("rates"), dict):  # alternative shape
        rate = payload["rates"].get("KES")

    if rate is not None:
        cache[key] = float(rate)
    return rate


# Static fallback used when the FX API is unreachable. Values are *approximate*
# monthly averages from CBK quarterly bulletins / publicly reported figures and
# are accurate enough to use as an exogenous regressor for a student project.
FX_FALLBACK = {
    # 2022
    "2022-05": 116.5, "2022-08": 119.5, "2022-11": 122.5,
    # 2023
    "2023-02": 126.5, "2023-05": 137.5, "2023-08": 144.5, "2023-11": 152.5,
    # 2024
    "2024-01": 159.5, "2024-02": 156.5, "2024-03": 132.0, "2024-04": 131.0,
    "2024-05": 131.5, "2024-06": 128.5, "2024-07": 129.0, "2024-08": 129.0,
    "2024-09": 129.0, "2024-10": 129.0, "2024-11": 129.0, "2024-12": 129.0,
    # 2025
    "2025-01": 129.5, "2025-02": 129.5, "2025-03": 129.5, "2025-04": 129.5,
    "2025-05": 129.5, "2025-06": 129.0, "2025-07": 129.0, "2025-08": 129.0,
    "2025-09": 129.0, "2025-10": 129.0, "2025-11": 129.0, "2025-12": 129.0,
    # 2026
    "2026-01": 129.0, "2026-02": 129.0, "2026-03": 129.0, "2026-04": 129.0,
    "2026-05": 129.0,
}


def get_fx_for_month(period: date, cache: dict[str, float]) -> float | None:
    """Return a USD→KES rate for the given month, using cache → API → fallback."""
    rate = fetch_usd_kes(period, cache)
    if rate is not None:
        return rate
    return FX_FALLBACK.get(period.strftime("%Y-%m"))


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the extracted table but don't overwrite the CSV.")
    parser.add_argument("--skip-fx", action="store_true",
                        help="Don't fetch USD/KES rates; leave the column NaN.")
    args = parser.parse_args()

    if not PDF_DIR.exists():
        log.error("PDF directory %s does not exist.", PDF_DIR)
        sys.exit(1)

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    log.info("Found %d PDF files in %s", len(pdfs), PDF_DIR)

    rows = []
    skipped = []
    for pdf_path in pdfs:
        row = extract_monthly_pdf(pdf_path)
        if row is None:
            skipped.append(pdf_path.name)
            continue
        rows.append(row)
        log.info("  ✓ %-25s → %s  (BP1 pkgs=%s, BP1 avg=%s)",
                 pdf_path.name,
                 row["date"].strftime("%b %Y"),
                 row.get("BP1_pkgs"), row.get("BP1"))

    if not rows:
        log.error("No PDF rows extracted; aborting.")
        sys.exit(1)

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Deduplicate by auction_no — the PDFs sometimes have the same auction
    # in two differently-named files (e.g. May 2023.pdf = May 2025.pdf,
    # both header "Auction Nos 2025/21"). Keep the row whose filename agrees
    # with the parsed auction year; otherwise keep the first.
    #
    # IMPORTANT: only dedup rows that actually have an auction_no. Rows with
    # a missing auction_no (e.g. daily-sale PDFs where OCR couldn't read the
    # header) must NOT be collapsed together — they are distinct sale dates.
    # ------------------------------------------------------------------
    has_no = df["auction_no"].astype(str).str.strip() != ""
    with_no    = df[has_no].copy()
    without_no = df[~has_no].copy()

    if with_no["auction_no"].duplicated().any():
        dupes = with_no[with_no["auction_no"].duplicated(keep=False)]
        log.warning("Duplicate auction_no found across %d rows:", len(dupes))
        for _, r in dupes.iterrows():
            log.warning("    auction %s  date %s  from %s",
                        r["auction_no"], r["date"].strftime("%Y-%m"),
                        r["source_pdf"])

        def _date_matches_auction(row):
            try:
                year = int(row["auction_no"].split("/")[0])
                return year == row["date"].year
            except (ValueError, AttributeError):
                return False

        with_no["_matches"] = with_no.apply(_date_matches_auction, axis=1)
        with_no = (with_no.sort_values(["auction_no", "_matches"],
                                       ascending=[True, False])
                          .drop_duplicates(subset="auction_no", keep="first")
                          .drop(columns="_matches"))

    df = (pd.concat([with_no, without_no], ignore_index=True)
            .sort_values("date")
            .reset_index(drop=True))
    log.info("After dedup: %d rows (%d with auction_no, %d without).",
             len(df), len(with_no), len(without_no))

    # Add USD/KES
    if not args.skip_fx:
        cache = _load_fx_cache()
        fx_values = []
        for ts in df["date"]:
            r = get_fx_for_month(ts.date(), cache)
            fx_values.append(r)
            time.sleep(0.1)
        df["usd_kes"] = fx_values
        _save_fx_cache(cache)
        log.info("FX rates added (cache entries: %d)", len(cache))
    else:
        df["usd_kes"] = pd.NA

    # Final column order
    base_cols   = ["date", "auction_no"]
    price_cols  = list(GRADE_TO_CSV.values())
    volume_cols = [f"{g}_pkgs" for g in GRADE_TO_CSV.values()]
    extra_cols  = ["total_pkgs", "total_kgs", "total_avg", "usd_kes", "source_pdf"]
    df = df[base_cols + price_cols + volume_cols + extra_cols]

    # Pretty-print
    print()
    print("Extracted rows:")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.to_string(index=False))

    if args.dry_run:
        log.info("Dry run — CSV not written.")
    else:
        # Format date to ISO (so it round-trips with the existing parser).
        out = df.copy()
        out["date"] = out["date"].dt.strftime("%Y-%m-%d")
        CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(CSV_PATH, index=False)
        log.info("✓ Wrote %s  (%d rows, %d cols)",
                 CSV_PATH.relative_to(ROOT), len(out), len(out.columns))

    if skipped:
        log.info("Skipped %d PDFs (no extractable text — likely daily image-only files):",
                 len(skipped))
        for name in skipped:
            log.info("    - %s", name)


if __name__ == "__main__":
    main()
