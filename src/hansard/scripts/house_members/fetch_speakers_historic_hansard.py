import re, time, json, os
from pathlib import Path
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

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
def get(url, allow_404=False, timeout_connect=5, timeout_read=10, max_attempts=2):
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
def discover_letters():
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
    # normalize unicode dashes/spaces and collapse whitespace
    return (
        s.replace("\u2013", "-")   # en dash –
         .replace("\u2014", "-")   # em dash —
         .replace("\u2212", "-")   # minus sign −
         .replace("\xa0", " ")     # NBSP
    )

def parse_lifespan(text: str):
    """
    Robustly extract (birth_year, death_year) from strings like:
      'June 30, 1962 –'            -> ('1962', None)
      '– August 14, 1863'          -> (None, '1863')
      '1891 – June 12, 1963'       -> ('1891', '1963')
      '1853 – 1895'                -> ('1853', '1895')
    Falls back to single year as birth if no dash is present.
    """
    t = _norm(text or "")
    t = re.sub(r"\s+", " ", t).strip()

    # collect all 4-digit years in order (restrict to plausible range)
    years = re.findall(r"(?<!\d)(1[6-9]\d{2}|20\d{2})(?!\d)", t)
    if not years:
        return (None, None)

    if len(years) >= 2:
        return (years[0], years[-1])

    # single year: decide side using dash position
    has_dash = "-" in t
    if has_dash:
        dash_pos = t.find("-")
        year_pos = t.find(years[0])
        if dash_pos <= year_pos:
            # dash before the year -> only death is present
            return (None, years[0])
        else:
            # dash after the year -> only birth is present
            return (years[0], None)

    # no dash at all; assume it's a lone birth year from a partial record
    return (years[0], None)


def parse_letter_page(letter, debug=False):
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

    # Dedup within letter by slug
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
def fetch_person_json(person_slug):
    url_js = f"{BASE}/people/{person_slug}.js"
    r = get(url_js)
    txt = r.text.strip()
    if not (txt.startswith("{") and txt.endswith("}")):
        raise ValueError("Non-JSON response at .js endpoint")
    return r.json()

def _iter_section_items(soup, header_contains: str):
    """
    Find the section whose heading text contains `header_contains` (case-insensitive),
    then yield all <li> items until the next h2/h3.
    """
    h = soup.find(lambda tag: tag.name in ("h2", "h3") and header_contains.lower() in tag.get_text(" ", strip=True).lower())
    if not h:
        return []

    items = []
    for sib in h.find_next_siblings():
        if getattr(sib, "name", None) in ("h2", "h3"):
            break
        for li in sib.find_all("li"):
            # Keep both the full text and (optionally) the link text/href if present
            a = li.find("a", href=True)
            full_text = li.get_text(" ", strip=True)
            item = {"text": full_text}
            if a:
                item["link_text"] = a.get_text(strip=True)
                item["href"] = urljoin(BASE, a["href"])
            items.append(item)
    return items

def scrape_person_html(person_slug):
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
        "titles_in_lords": titles_lords,   # <- new field; safe to be empty list
    }

def enrich_row(row):
    """Fetch page details for one person row. Resilient; never raises."""
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
            d = scrape_person_html(row["slug"])
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
    throttle=0.0,
    fetch_details=False,       # FAST MODE by default
    max_per_letter=None,       # e.g., 20 for a smoke test
    concurrency=0,             # set to 8..16 only when fetch_details=True
    delete_checkpoints=True,   # <-- NEW: delete per-letter checkpoints at the end
    debug=False
):
    letters = discover_letters()
    print("Letters discovered:", letters)

    ckpt_paths = []
    all_chunks = []

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
                        if done % 25 == 0:
                            print(f"  {letter.upper()}: fetched {done}/{len(rows)}")
                        if throttle:
                            time.sleep(throttle)
            else:
                for i, r in enumerate(rows, 1):
                    chunk.append(enrich_row(r))
                    if i % 25 == 0:
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
    
    # Delete unnecessary columns
    df_all = df_all.drop(columns=["page_title", "birth_year", "death_year", "source_format", "error"], errors="ignore")
    df_all.to_parquet(OUT_PATH, index=False)
    print(f"Wrote {len(df_all):,} people to {OUT_PATH}")

    # -------- Cleanup checkpoints at the very end --------
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
    # For details later:
    main(fetch_details=True, max_per_letter=None, concurrency=8, delete_checkpoints=True)
