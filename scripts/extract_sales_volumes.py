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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("extract")

ROOT       = Path(__file__).resolve().parent.parent
PDF_DIR    = ROOT / "salesreport"
CSV_PATH   = ROOT / "analytics" / "data" / "tea_auction_data.csv"
FX_CACHE   = ROOT / "analytics" / "data" / ".fx_cache.json"

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
AUCTION_HEADER_RE = re.compile(r"Auction Nos\s+(\d{4})/(\d{1,3})\s+to\s+(\d{4})/(\d{1,3})")


def _to_number(token: str) -> float | None:
    if not NUM_RE.match(token):
        return None
    try:
        return float(token.replace(",", ""))
    except ValueError:
        return None


def filename_to_period(name: str) -> date | None:
    """
    Resolve "Jan 2025.pdf", "May 2023.pdf", etc. to a month-start date.
    Daily filenames like "sale for 13.03.2026.pdf" are NOT handled here —
    those PDFs have no extractable text and are skipped upstream.
    """
    stem = Path(name).stem.lower().strip()
    if stem.startswith("sale for"):
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


def extract_monthly_pdf(path: Path) -> dict | None:
    """
    Extract the period + per-grade Pkgs/Avg + totals from a monthly PDF.
    Returns a dict suitable for one CSV row.
    """
    period = filename_to_period(path.name)
    if period is None:
        log.warning("Skipping %s — could not parse period from filename.", path.name)
        return None

    try:
        pdf = pdfplumber.open(str(path))
    except Exception as e:
        log.error("Failed to open %s: %s", path.name, e)
        return None

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
                # Totals row: every token must be numeric & at least 5 of them.
                nums = [_to_number(t) for t in tokens]
                if all(n is not None for n in nums) and len(nums) >= 7:
                    totals_numbers = [n for n in nums if n is not None]  # type: ignore[misc]
                    # Don't break — keep the LAST numeric-only row, which is
                    # always the bottom totals on the final page.
    finally:
        pdf.close()

    if totals_numbers is None:
        log.warning("Could not locate totals row in %s", path.name)
        return None

    parsed = parse_totals_row(totals_numbers)
    if parsed is None:
        log.warning("Failed to parse totals row in %s: %s", path.name, totals_numbers)
        return None

    row = {
        "date":       pd.Timestamp(period),
        "auction_no": auction_no or "",
        # Per-grade price (Avg from the totals row)
        **{g: parsed.get(f"{g}_avg") for g in GRADE_TO_CSV.values()},
        # Per-grade sales volume (Pkgs)
        **{f"{g}_pkgs": parsed.get(f"{g}_pkgs") for g in GRADE_TO_CSV.values()},
        # Aggregate volumes for the month
        "total_pkgs": parsed.get("total_pkgs"),
        "total_kgs":  parsed.get("total_kgs"),
        "total_avg":  parsed.get("total_avg"),
        "source_pdf": path.name,
    }
    return row


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

    # Detect duplicates by auction_no — the PDFs sometimes have the same
    # auction in two differently-named files (e.g. May 2023.pdf = May 2025.pdf,
    # both header "Auction Nos 2025/21"). Keep the row whose filename agrees
    # with the parsed auction year; otherwise keep the first.
    if df["auction_no"].duplicated().any():
        dupes = df[df["auction_no"].duplicated(keep=False)]
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

        df["_matches"] = df.apply(_date_matches_auction, axis=1)
        # Keep rows that match by year; fall back to first occurrence.
        df = (df.sort_values(["auction_no", "_matches"], ascending=[True, False])
                .drop_duplicates(subset="auction_no", keep="first")
                .drop(columns="_matches")
                .sort_values("date")
                .reset_index(drop=True))
        log.info("After dedup: %d rows.", len(df))

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
