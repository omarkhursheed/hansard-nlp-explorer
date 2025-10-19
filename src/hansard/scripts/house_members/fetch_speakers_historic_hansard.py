"""
Historic Hansard People Crawler

This script scrapes the “People” index from the historic Hansard website and
builds a parquet of speakers with lightweight metadata. It supports:

- Robust letter-page discovery (handles missing letters like X)
- Parsing list-level birth/death years (incl. en/em dashes and one-sided spans)
- Optional per-person detail fetch (JSON endpoint → HTML fallback)
- Section-bounded scraping for “Constituencies” and “Titles in Lords”
- Per-letter checkpoints (and optional cleanup)
- Fast mode (no detail fetch) and threaded detail fetch for speed

Typical usage:
    # Fast sanity run (no per-person fetch)
    # main(fetch_details=False, max_per_letter=None, concurrency=0, delete_checkpoints=True)

    # Full detail run with concurrency
    # main(fetch_details=True, max_per_letter=None, concurrency=8, delete_checkpoints=True)
"""

import re
import time
import json
from pathlib import Path
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup
import pandas as pd

BASE = "https://api.parliament.uk/historic-hansard"
HEADERS = {"User-Agent": "HansardResearchBot/1.1 (academic; contact: your.email@example.com)"}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)

OUT_DIR = Path("src/hansard/data/processed_fixed/metadata/house_members")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "historic_hansard_speakers.parquet"

LETTER_RE = re.compile(r"/people/([A-Za-z])/?" + r"(?:$|[?#])")
SLUG_RE   = re.compile(r"/people/([^/?#]+)/?")
NAV_LETTER_PATH_RE = re.compile(r"^/historic-hansard/people/([a-z]|index\.html)/?$")


# ----------------------------
# HTTP (short timeouts; fewer retries)
# ----------------------------
def get(url: str, allow_404: bool = False, timeout_connect: int = 5, timeout_read: int = 10, max_attempts: int = 2) -> requests.Response | None:
    """
    GET a URL with short timeouts and limited retries.

    Args:
        url: Absolute URL to fetch.
        allow_404: If True, return None on 404 instead of raising.
        timeout_connect: Connect timeout (seconds).
        timeout_read: Read timeout (seconds).
        max_attempts: Max retry attempts (429/backoff, transient errors).

    Returns:
        A requests.Response on success; None if allow_404=True and status is 404.

    Raises:
        The last requests.RequestException if all attempts fail and allow_404=False.
    """
    last_exc = None
    for attempt in range(max_attempts):
        try:
            r = SESSION.get(url, timeout=(timeout_connect, timeout_read))
            if r.status_code == 404 and allow_404:
                return None
            if r.status_code == 429:
                time.sleep(1.0 * (2 ** attempt))
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            last_exc = e
            time.sleep(0.6 * (attempt + 1))
    if last_exc:
        raise last_exc


# ----------------------------
# Letter discovery
# ----------------------------
def discover_letters() -> list[str]:
    """
    Discover which A–Z letter pages exist for People.

    Strategy:
        1) Parse links off /people/index.html (A page).
        2) If too few discovered, probe /people/b..z with allow_404.

    Returns:
        Sorted list of one-letter strings (e.g., ['a','b','c',...]).
    """
    letters = {"a"}  # 'A' is index.html
    idx_url = f"{BASE}/people/index.html"
    r = get(idx_url)
    soup = BeautifulSoup(r.text, "html.parser")

    for a in soup.find_all("a", href=True):
        m = LETTER_RE.search(a["href"])
        if m:
            letters.add(m.group(1).lower())

    # Probe if parsing missed some
    if len(letters) < 5:
        for ch in "bcdefghijklmnopqrstuvwxyz":
            if get(f"{BASE}/people/{ch}", allow_404=True) is not None:
                letters.add(ch)

    return sorted(letters)


# ----------------------------
# Parsing helpers
# ----------------------------
def _norm(s: str) -> str:
    """
    Normalize unicode punctuation and spaces for lifespan parsing.

    Replaces:
        –, —, − → '-'
        NBSP    → ' '
    """
    return (
        s.replace("\u2013", "-")   # en dash –
         .replace("\u2014", "-")   # em dash —
         .replace("\u2212", "-")   # minus sign −
         .replace("\xa0", " ")     # NBSP
    )


def parse_lifespan(text: str) -> tuple[str | None, str | None]:
    """
    Extract (birth_year, death_year) from list/person-page text.

    Handles:
        'June 30, 1962 –'            -> ('1962', None)
        '– August 14, 1863'          -> (None, '1863')
        '1891 – June 12, 1963'       -> ('1891', '1963')
        '1853 – 1895'                -> ('1853', '1895')
        '1934'                       -> ('1934', None)

    Args:
        text: Raw text fragment containing life dates (possibly one-sided).

    Returns:
        Tuple (birth_year, death_year) as strings or None where missing.
    """
    t = _norm(text or "")
    t = re.sub(r"\s+", " ", t).strip()

    # collect all plausible 4-digit years in order
    years = re.findall(r"(?<!\d)(1[6-9]\d{2}|20\d{2})(?!\d)", t)
    if not years:
        return (None, None)

    if len(years) >= 2:
        return (years[0], years[-1])

    # single year: infer side by dash position
    has_dash = "-" in t
    if has_dash:
        dash_pos = t.find("-")
        year_pos = t.find(years[0])
        if dash_pos <= year_pos:
            return (None, years[0])   # dash before year → only death present
        else:
            return (years[0], None)   # dash after year → only birth present

    # no dash; assume it's a lone birth year
    return (years[0], None)


def parse_letter_page(letter: str, debug: bool = False) -> list[dict]:
    """
    Parse one letter page and return list entries.

    Args:
        letter: Lowercase letter (e.g., 'a', 'b', ...). 'a' maps to index.html.
        debug: If True, prints sample hrefs discovered.

    Returns:
        List of dicts:
            {
              "name": str,
              "slug": str,
              "list_birth_year": str|None,
              "list_death_year": str|None
            }
    """
    page_url = f"{BASE}/people/index.html" if letter == "a" else f"{BASE}/people/{letter}"
    soup = BeautifulSoup(get(page_url).text, "html.parser")

    anchors = soup.select("li a[href]")
    if debug:
        print(f"debug: {letter} -> {len(anchors)} li>a total; sample hrefs:",
              [a.get("href") for a in anchors[:3]])

    out = []
    for a in anchors:
        href = a.get("href", "")
        abs_url = urljoin(page_url, href)
        path = urlparse(abs_url).path

        # Skip navbar/letter/self links like /people/b or /people/index.html
        if NAV_LETTER_PATH_RE.match(path):
            continue
        if "/historic-hansard/people/" not in path:
            continue

        m = SLUG_RE.search(path)
        if not m:
            continue
        slug = m.group(1).strip("/")
        name = a.get_text(strip=True)

        li = a.find_parent("li")
        remainder = li.get_text(" ", strip=True) if li else ""
        if name and remainder.startswith(name):
            remainder = remainder[len(name):].strip()
        birth, death = parse_lifespan(remainder)

        out.append({"name": name, "slug": slug, "list_birth_year": birth, "list_death_year": death})

    # Deduplicate within letter by slug
    seen = set()
    uniq = []
    for r in out:
        if r["slug"] not in seen:
            seen.add(r["slug"])
            uniq.append(r)
    return uniq


# ----------------------------
# Detail fetch (JSON → HTML fallback)
# ----------------------------
def fetch_person_json(person_slug: str) -> dict:
    """
    Attempt to fetch a person's page via the '.js' JSON endpoint.

    Args:
        person_slug: Path fragment after '/people/'.

    Returns:
        Parsed JSON dict (raises ValueError if the response is not proper JSON).
    """
    url_js = f"{BASE}/people/{person_slug}.js"
    r = get(url_js)
    txt = r.text.strip()
    if not (txt.startswith("{") and txt.endswith("}")):
        raise ValueError("Non-JSON response at .js endpoint")
    return r.json()


def _iter_section_items(soup: BeautifulSoup, header_contains: str) -> list[dict]:
    """
    Yield list items under a specific section until the next h2/h3.

    Args:
        soup: Parsed BeautifulSoup document for a person page.
        header_contains: Case-insensitive substring of the target section header
                         (e.g., 'Constituencies', 'Titles in Lords').

    Returns:
        List of dicts per <li>:
            { "text": full_text, "link_text": optional, "href": optional }
    """
    h = soup.find(
        lambda tag: tag.name in ("h2", "h3")
        and header_contains.lower() in tag.get_text(" ", strip=True).lower()
    )
    if not h:
        return []

    items = []
    for sib in h.find_next_siblings():
        if getattr(sib, "name", None) in ("h2", "h3"):
            break
        for li in sib.find_all("li"):
            a = li.find("a", href=True)
            full_text = li.get_text(" ", strip=True)
            item = {"text": full_text}
            if a:
                item["link_text"] = a.get_text(strip=True)
                item["href"] = urljoin(BASE, a["href"])
            items.append(item)
    return items


def scrape_person_html(person_slug: str) -> dict:
    """
    HTML fallback: scrape a person's page for title, life dates, constituencies,
    and “Titles in Lords” section (bounded to section to avoid bleed-through).

    Args:
        person_slug: Path fragment after '/people/'.

    Returns:
        Dict with keys:
            {
              "title": str|None,
              "birth_year": str|None,
              "death_year": str|None,
              "constituencies": list[{"constituency": str|None, "from_to": str}],
              "titles_in_lords": list[str]
            }
    """
    url = f"{BASE}/people/{person_slug}"
    soup = BeautifulSoup(get(url).text, "html.parser")

    # Title (name) and dates line
    h1 = soup.find("h1")
    title = (h1.get_text(strip=True) if h1 else None)
    dates_text = h1.find_next(string=True) if h1 else None
    birth, death = parse_lifespan((dates_text or "").strip())

    # Constituencies (bounded to its section)
    const = []
    for item in _iter_section_items(soup, "Constituencies"):
        link_txt = item.get("link_text")
        txt = item["text"]
        if link_txt and txt.startswith(link_txt):
            txt = txt[len(link_txt):].strip()
        const.append({
            "constituency": link_txt or None,
            "from_to": txt
        })

    # Titles in Lords (bounded to its section) — free-text list
    titles_lords = [it["text"] for it in _iter_section_items(soup, "Titles in Lords")]

    return {
        "title": title,
        "birth_year": birth,
        "death_year": death,
        "constituencies": const,
        "titles_in_lords": titles_lords,
    }


def enrich_row(row: dict) -> dict:
    """
    Enrich one row (from a letter list) with per-person details.

    Tries the JSON endpoint first; falls back to HTML scraper.
    Never raises: on repeated failure, emits a minimal row with 'source_format=error'.

    Args:
        row: Dict with list-level fields (name, slug, list_birth_year, list_death_year).

    Returns:
        A merged dict with URL, list fields, and detail fields (birth/death/constituencies, etc.).
    """
    slug = row["slug"]
    details = {}
    try:
        j = fetch_person_json(slug)
        title = j.get("title") or j.get("person", {}).get("title")
        birth = j.get("birth") or j.get("birth_year")
        death = j.get("death") or j.get("death_year")
        consts = j.get("constituencies") or []
        details = {
            "page_title": title or row["name"],
            "birth_year": birth,
            "death_year": death,
            "constituencies": json.dumps(consts, ensure_ascii=False),
            "source_format": "json",
        }
    except Exception:
        try:
            d = scrape_person_html(slug)
            details = {
                "page_title": d.get("title") or row["name"],
                "birth_year": d.get("birth_year"),
                "death_year": d.get("death_year"),
                "constituencies": json.dumps(d.get("constituencies", []), ensure_ascii=False),
                "titles_in_lords": json.dumps(d.get("titles_in_lords", []), ensure_ascii=False),
                "source_format": "html",
            }
        except Exception as e2:
            details = {
                "page_title": row["name"],
                "birth_year": None,
                "death_year": None,
                "constituencies": "[]",
                "titles_in_lords": "[]",
                "source_format": "error",
                "error": str(e2),
            }
    return {
        "name": row["name"],
        "slug": row["slug"],
        "url": f"{BASE}/people/{row['slug']}",
        "list_birth_year": row["list_birth_year"],
        "list_death_year": row["list_death_year"],
        **details,
    }


# ----------------------------
# Main
# ----------------------------
def main(
    throttle: float = 0.0,
    fetch_details: bool = False,
    max_per_letter: int | None = None,
    concurrency: int = 0,
    delete_checkpoints: bool = True,
    debug: bool = False
) -> None:
    """
    Crawl the People index and write a combined parquet.

    Args:
        throttle: Sleep (seconds) between person fetches (detail mode).
        fetch_details: If True, fetch per-person details; else, list-only fast mode.
        max_per_letter: Limit the number of people processed per letter (None for all).
        concurrency: Thread count for detail fetches (ignored if fetch_details=False).
        delete_checkpoints: If True, delete per-letter parquet checkpoints at end.
        debug: If True, prints sampling info from letter pages.

    Side effects:
        - Writes per-letter checkpoints to OUT_DIR.
        - Writes combined parquet to OUT_PATH.
        - Optionally deletes checkpoints.
    """
    letters = discover_letters()
    print("Letters discovered:", letters)

    ckpt_paths: list[Path] = []
    all_chunks: list[pd.DataFrame] = []

    for letter in letters:
        rows = parse_letter_page(letter, debug=debug)
        if max_per_letter:
            rows = rows[:max_per_letter]
        print(f"[{letter.upper()}] rows: {len(rows)}")

        if not rows:
            continue

        if not fetch_details:
            # FAST: no per-person hits
            chunk = [
                {
                    "name": r["name"],
                    "slug": r["slug"],
                    "url": f"{BASE}/people/{r['slug']}",
                    "list_birth_year": r["list_birth_year"],
                    "list_death_year": r["list_death_year"],
                    "page_title": r["name"],
                    "birth_year": r["list_birth_year"],
                    "death_year": r["list_death_year"],
                    "constituencies": "[]",
                    "titles_in_lords": "[]",
                    "source_format": "list_only",
                }
                for r in rows
            ]
        else:
            # DETAIL MODE: sequential or threaded
            chunk = []
            if concurrency and concurrency > 0:
                with ThreadPoolExecutor(max_workers=concurrency) as ex:
                    futs = [ex.submit(enrich_row, r) for r in rows]
                    done = 0
                    for fut in as_completed(futs):
                        chunk.append(fut.result())
                        done += 1
                        if done % 25 == 0 or done == len(rows):
                            print(f"  {letter.upper()}: fetched {done}/{len(rows)}")
                        if throttle:
                            time.sleep(throttle)
            else:
                for i, r in enumerate(rows, 1):
                    chunk.append(enrich_row(r))
                    if i % 25 == 0 or i == len(rows):
                        print(f"  {letter.upper()}: fetched {i}/{len(rows)}")
                    if throttle:
                        time.sleep(throttle)

        df_chunk = pd.DataFrame(chunk).drop_duplicates(subset=["slug"])
        ckpt_path = OUT_DIR / f"historic_hansard_speakers_{letter}.parquet"
        df_chunk.to_parquet(ckpt_path, index=False)
        ckpt_paths.append(ckpt_path)
        print(f"  saved checkpoint: {ckpt_path} ({len(df_chunk)} rows)")
        all_chunks.append(df_chunk)

    if not all_chunks:
        print("No rows collected.")
        return

    df_all = pd.concat(all_chunks, ignore_index=True).drop_duplicates(subset=["slug"])

    # Drop transient columns you don't want in the final output
    df_all = df_all.drop(columns=["page_title", "birth_year", "death_year", "source_format", "error"], errors="ignore")
    df_all.to_parquet(OUT_PATH, index=False)
    print(f"Wrote {len(df_all):,} people to {OUT_PATH}")

    # Cleanup checkpoints at the very end
    if delete_checkpoints:
        removed = 0
        for p in ckpt_paths:
            try:
                p.unlink(missing_ok=True)
                removed += 1
            except Exception as e:
                print(f"  warn: could not delete {p}: {e}")
        print(f"Deleted {removed}/{len(ckpt_paths)} checkpoint files.")


if __name__ == "__main__":
    # Fast sanity run:
    # main(fetch_details=False, max_per_letter=None, concurrency=0, delete_checkpoints=True, debug=False)

    # Full detail run:
    main(fetch_details=True, max_per_letter=None, concurrency=8, delete_checkpoints=True)
