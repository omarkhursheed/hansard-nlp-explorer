from __future__ import annotations
"""Historic Hansard crawler **v4.2** – production‑ready, disk‑efficient, and notebook‑friendly.

Changes vs v4.1 (2025‑07‑12)
────────────────────────────
✓ **FIXED API navigation** - properly handles 3-level hierarchy (decade → year → month → day)
✓ **Improved error handling** throughout the crawler
✓ **Better regex patterns** for link extraction
✓ **Enhanced path validation** in save method
✓ **Optimized batch processing** with better timeout handling
✓ **Added progress tracking** and better logging
✓ **Fixed potential encoding issues** in file operations
✓ **Added data validation** to prevent corrupted saves

The API structure is: /sittings/1860s → /sittings/1864 → /sittings/1864/feb → /sittings/1864/feb/15

Usage examples are unchanged; see `python crawler.py --help`.
"""

import argparse
import asyncio
import gzip
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ───────────────────── Optional fast HTML parser ─────────────────────
try:
    from selectolax.parser import HTMLParser as _HTML
    _USE_SELECTOLAX = True
except ImportError:  # fallback
    from bs4 import BeautifulSoup as _BS
    _HTML = None  # type: ignore
    _USE_SELECTOLAX = False

# ───────────────────────── Env helpers ───────────────────────────────
try:
    import nest_asyncio  # notebook loop patch
except ImportError:
    nest_asyncio = None  # type: ignore

BASE_URL = "https://api.parliament.uk/historic-hansard"
DEFAULT_CONCURRENCY = 4
MAX_RPS = 3  # be more conservative 

# Fast batch processing settings
FAST_CONCURRENCY = 8
FAST_MAX_RPS = 6.0 
MONTHS = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
FULL_MONTHS = [
    "january","february","march","april","may","june",
    "july","august","september","october","november","december",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s – %(levelname)s – %(message)s")
log = logging.getLogger("hansard")

# ────────────────────── Rate‑limited semaphore ───────────────────────
class _Limiter:
    def __init__(self, concurrent: int, rps: float):
        self.sem = asyncio.Semaphore(concurrent)
        self.min_int = 1.0 / rps
        self._t_last = 0.0
    
    async def __aenter__(self):
        await self.sem.acquire()
        await self._pace()
        return self
    
    async def __aexit__(self, *_):
        self.sem.release()
    
    async def _pace(self):
        loop = asyncio.get_event_loop()
        now = loop.time()
        elapsed = now - self._t_last
        if elapsed < self.min_int:
            sleep_time = self.min_int - elapsed
            await asyncio.sleep(sleep_time)
        self._t_last = loop.time()

# ───────────────────────── Crawler class ─────────────────────────────
class HansardCrawler:
    def __init__(self, concurrency: int = DEFAULT_CONCURRENCY):
        self.concurrency = concurrency
        self.http: Optional[httpx.AsyncClient] = None
        self.limiter: Optional[_Limiter] = None
        self.processed_count = 0

    async def __aenter__(self):
        self.http = httpx.AsyncClient(
            http2=True, 
            follow_redirects=True,
            timeout=30.0,
            headers={"User-Agent": "HansardCrawler/4.2"},
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)
        )
        self.limiter = _Limiter(self.concurrency, MAX_RPS)
        return self

    async def __aexit__(self, *_):
        if self.http:
            await self.http.aclose()

    # ─────────────── Fetch helper with retry & 404 swallow ───────────
    @retry(wait=wait_exponential(multiplier=1, min=2, max=8),
           stop=stop_after_attempt(3),
           retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.RequestError)))
    async def _get(self, url: str, expect_json: bool = False):
        assert self.http and self.limiter
        try:
            async with self.limiter:
                r = await self.http.get(url)
                if r.status_code == 404:
                    log.debug(f"404 Not Found: {url}")
                    return None
                r.raise_for_status()
                return r.json() if expect_json else r.text
        except Exception as e:
            log.debug(f"Failed to fetch {url}: {e}")
            return None

    # ─────────────── Extract links from HTML ──────────────
    def _extract_links(self, html: str) -> List[str]:
        """Extract all href links from HTML."""
        try:
            parser = (_HTML(html) if _USE_SELECTOLAX else _BS(html, "html.parser"))
            
            if _USE_SELECTOLAX:
                anchors = parser.css("a")
                return [a.attributes.get("href", "") for a in anchors if a.attributes.get("href")]
            else:
                anchors = parser.find_all("a", href=True)
                return [a.get("href", "") for a in anchors]
        except Exception as e:
            log.debug(f"Error extracting links: {e}")
            return []

    # ─────────────── Discovery: Decade → Years ──────────────
    async def discover_years_in_decade(self, decade: str) -> List[str]:
        """Get all years in a decade (e.g., '1860s' → ['1860', '1861', ...])."""
        try:
            html = await self._get(f"{BASE_URL}/sittings/{decade}")
            if not html:
                return []
            
            links = self._extract_links(html)
            years = []
            
            decade_start = int(decade[:-1])  # '1860s' → 1860
            decade_end = decade_start + 9
            
            for link in links:
                # Look for year patterns like '/sittings/1864' or just '1864'
                if link.isdigit() and len(link) == 4:
                    year = int(link)
                    if decade_start <= year <= decade_end:
                        years.append(link)
                elif link.startswith('/historic-hansard/sittings/'):
                    year_part = link.split('/')[-1]
                    if year_part.isdigit() and len(year_part) == 4:
                        year = int(year_part)
                        if decade_start <= year <= decade_end:
                            years.append(year_part)
            
            return sorted(set(years))
        except Exception as e:
            log.error(f"Error discovering years in decade {decade}: {e}")
            return []

    # ─────────────── Discovery: Year → Months ──────────────
    async def discover_months_in_year(self, year: str) -> List[str]:
        """Get all months in a year (e.g., '1864' → ['1864/feb', '1864/mar', ...])."""
        try:
            html = await self._get(f"{BASE_URL}/sittings/{year}")
            if not html:
                return []
            
            links = self._extract_links(html)
            months = []
            
            for link in links:
                # Look for month patterns like '/historic-hansard/sittings/1864/feb'
                if f"/sittings/{year}/" in link:
                    # Extract the month part
                    parts = link.split(f"/sittings/{year}/")
                    if len(parts) > 1:
                        month_part = parts[1].split('/')[0]  # Get first part after year
                        if month_part.lower() in MONTHS or month_part.lower() in [m[:3] for m in FULL_MONTHS]:
                            months.append(f"{year}/{month_part}")
            
            return sorted(set(months))
        except Exception as e:
            log.error(f"Error discovering months in year {year}: {e}")
            return []

    # ─────────────── Discovery: Month → Days ──────────────
    async def discover_days_in_month(self, year_month: str) -> List[str]:
        """Get all sitting days in a month (e.g., '1864/feb' → ['1864/feb/15', '1864/feb/16', ...])."""
        try:
            html = await self._get(f"{BASE_URL}/sittings/{year_month}")
            if not html:
                log.warning(f"No HTML returned for {year_month}")
                return []
            
            links = self._extract_links(html)
            days = []
            
            log.debug(f"Found {len(links)} links for {year_month}")
            
            for link in links:
                # Look for day patterns like '/historic-hansard/sittings/1864/feb/15'
                if f"/sittings/{year_month}/" in link:
                    # Extract the day part
                    parts = link.split(f"/sittings/{year_month}/")
                    if len(parts) > 1:
                        day_part = parts[1].split('/')[0]  # Get first part after month
                        if day_part.isdigit():
                            days.append(f"{year_month}/{day_part}")
            
            # Add explicit logging for single-digit days discovery
            single_digit_days = [d for d in days if d.split('/')[-1].isdigit() and len(d.split('/')[-1]) == 1]
            if single_digit_days:
                log.info(f"Found single-digit days in {year_month}: {single_digit_days}")
            
            result = sorted(set(days))
            log.debug(f"Discovered {len(result)} days in {year_month}: {result}")
            return result
        except Exception as e:
            log.error(f"Error discovering days in month {year_month}: {e}")
            return []

    # ─────────────── Complete Discovery: Decade → All Days ──────────────
    async def discover_all_sitting_days(self, decade: str, year_filter: Optional[Set[int]] = None) -> List[str]:
        """Full hierarchy traversal: decade → years → months → days."""
        try:
            log.info(f"Discovering years in {decade}...")
            years = await self.discover_years_in_decade(decade)
            
            # Filter years if specified
            if year_filter:
                years = [y for y in years if int(y) in year_filter]
            
            log.info(f"Found {len(years)} years to process: {years}")
            
            if not years:
                return []
            
            all_days = []
            
            for year in years:
                log.debug(f"Discovering months in {year}...")
                months = await self.discover_months_in_year(year)
                log.debug(f"Found {len(months)} months in {year}: {months}")
                
                for month in months:
                    log.debug(f"Discovering days in {month}...")
                    days = await self.discover_days_in_month(month)
                    log.debug(f"Found {len(days)} days in {month}")
                    all_days.extend(days)
            
            return sorted(set(all_days))
        except Exception as e:
            log.error(f"Error in complete discovery for {decade}: {e}")
            return []

    # ───────────── Debate link extractor ─────────────
    def _links(self, html: str, date_path: str) -> List[str]:
        """Extract debate links from a sitting day page."""
        try:
            parts = date_path.split("/")
            if len(parts) != 3:
                log.warning(f"Invalid date path format: {date_path}")
                return []
            
            y, mo, d = parts
            links = self._extract_links(html)
            out: List[str] = []
            
            # Handle both single-digit and zero-padded day formats
            # The API may redirect /7 to /07, so we need to check both
            d_padded = d.zfill(2) if d.isdigit() else d
            day_formats = [d, d_padded] if d != d_padded else [d]
            
            log.debug(f"Looking for debate links with day formats: {day_formats} in {date_path}")
            
            for link in links:
                if not link or any(x in link.lower() for x in ("index", "contents", "#")):
                    continue
                
                # Look for debate patterns with both day formats
                found_match = False
                for day_format in day_formats:
                    # Pattern 1: /commons/1864/feb/15/topic or /lords/1864/feb/15/topic
                    pattern1 = rf"/(commons|lords)/{re.escape(y)}/{re.escape(mo)}/{re.escape(day_format)}/([^/?#]+)"
                    m1 = re.search(pattern1, link, re.I)
                    if m1:
                        path = m1.group(0).lstrip("/")
                        if path not in out:
                            out.append(path)
                        found_match = True
                        break
                    
                    # Pattern 2: /1864/feb/15/topic (generic)
                    pattern2 = rf"/{re.escape(y)}/{re.escape(mo)}/{re.escape(day_format)}/([^/?#]+)"
                    m2 = re.search(pattern2, link, re.I)
                    if m2:
                        topic = m2.group(1)
                        # Add both commons and lords variants, but use original day format for consistency
                        for house in ["commons", "lords"]:
                            path = f"{house}/{y}/{mo}/{d}/{topic}"
                            if path not in out:
                                out.append(path)
                        found_match = True
                        break
                
                if found_match:
                    continue
            
            log.debug(f"Found {len(out)} debate links for {date_path}: {out[:5]}{'...' if len(out) > 5 else ''}")
            return out
        except Exception as e:
            log.error(f"Error extracting debate links from {date_path}: {e}")
            return []

    # ───────────── Debate fetch ─────────────
    async def _debate(self, path: str) -> Optional[Dict[str, Any]]:
        """Fetch a single debate."""
        try:
            html = await self._get(f"{BASE_URL}/{path}")
            if html and len(html.strip()) > 0:
                return {"path": path, "html": html}
            return None
        except Exception as e:
            log.debug(f"Error fetching debate {path}: {e}")
            return None

    # ───────────── One sitting day crawler ─────────────
    async def crawl_day(self, date_path: str, house: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Crawl all debates for a single sitting day."""
        try:
            # Add explicit logging for single-digit days
            day_num = date_path.split('/')[-1]
            if day_num.isdigit() and len(day_num) == 1:
                log.info(f"Processing single-digit day: {date_path}")
            
            html = await self._get(f"{BASE_URL}/sittings/{date_path}")
            if not html:
                log.warning(f"No HTML returned for sitting day: {date_path}")
                return None
            
            links = self._links(html, date_path)
            if house:
                links = [l for l in links if l.startswith(f"{house}/")]
            
            if not links:
                log.debug(f"No debate links found for {date_path}")
                return None
            
            # Fetch debates with limited concurrency
            tasks = [asyncio.create_task(self._debate(l)) for l in links]
            debates = []
            
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, Exception):
                        log.debug(f"Debate fetch failed: {result}")
                    elif result:
                        debates.append(result)
            except Exception as e:
                log.error(f"Error gathering debates for {date_path}: {e}")
            
            self.processed_count += 1
            if self.processed_count % 10 == 0:
                log.info(f"Processed {self.processed_count} sitting days...")
            
            return {
                "date": date_path,
                "debates": debates,
                "html": html[:4000]  # Keep first 4KB for reference
            }
        except Exception as e:
            log.error(f"Error crawling day {date_path}: {e}")
            return None

    # ───────────── Save (gzipped) ─────────────
    def _save(self, data: Dict[str, Any], outdir: Path) -> None:
        """Save crawled data to disk."""
        try:
            if not data or "date" not in data:
                log.warning("Invalid data structure, skipping save")
                return
            
            parts = data["date"].split("/")
            if len(parts) != 3:
                log.warning(f"Invalid date format: {data['date']}")
                return
            
            y, mo, d = parts
            
            # Add explicit logging for single-digit days
            if d.isdigit() and len(d) == 1:
                log.info(f"Saving single-digit day: {data['date']} with {len(data.get('debates', []))} debates")
            day_dir = outdir / y / mo
            day_dir.mkdir(parents=True, exist_ok=True)
            
            # Save summary JSON
            summary_path = day_dir / f"{d}_summary.json"
            summary_data = {
                "date": data["date"],
                "debate_count": len(data.get("debates", [])),
                "timestamp": datetime.now().isoformat()
            }
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            
            # Save debates
            for i, deb in enumerate(data.get("debates", [])):
                if not deb or "path" not in deb or "html" not in deb:
                    log.warning(f"Invalid debate data at index {i}")
                    continue
                
                # Create safe filename
                path_parts = deb["path"].split("/")
                if path_parts:
                    slug = re.sub(r"[^\w\-_.]", "_", path_parts[-1])
                    slug = re.sub(r"_+", "_", slug).strip("_")  # Clean up multiple underscores
                else:
                    slug = f"debate_{i}"
                
                filename = f"{d}_{i:02d}_{slug}.html.gz"
                filepath = day_dir / filename
                
                try:
                    compressed_data = gzip.compress(deb["html"].encode('utf-8'))
                    filepath.write_bytes(compressed_data)
                except Exception as e:
                    log.error(f"Failed to save debate {deb['path']}: {e}")
            
            log.debug(f"Saved {len(data.get('debates', []))} debates for {data['date']}")
            
        except Exception as e:
            log.error(f"Error saving data: {e}")

    # ───────────── Crawl range ─────────────
    async def crawl(self, start: int, end: int, out: Path, house: Optional[str], decade: bool) -> None:
        """Crawl a range of years."""
        try:
            # Get all decades needed
            decades_needed: Set[str] = set()
            for year in range(start, end + 1):
                decade_str = f"{(year // 10) * 10}s"
                decades_needed.add(decade_str)
            
            # Create year filter for efficiency
            year_filter = set(range(start, end + 1))
            
            # Discover all sitting days using proper hierarchy traversal
            all_days: List[str] = []
            for decade_str in sorted(decades_needed):
                log.info(f"Discovering sitting days in {decade_str} for years {start}-{end}...")
                decade_days = await self.discover_all_sitting_days(decade_str, year_filter)
                all_days.extend(decade_days)
            
            # Filter to requested year range (double-check)
            filtered_days = []
            for day in all_days:
                try:
                    year_str = day.split("/")[0]
                    if year_str.isdigit():
                        year = int(year_str)
                        if start <= year <= end:
                            filtered_days.append(day)
                except (IndexError, ValueError):
                    log.debug(f"Skipping malformed day path: {day}")
                    continue
            
            days = sorted(set(filtered_days))
            log.info(f"Found {len(days)} sitting days in range {start}-{end}")
            
            if not days:
                log.warning("No sitting days found in specified range")
                return
            
            # Process in small batches
            batch_size = 3
            total_batches = (len(days) + batch_size - 1) // batch_size
            
            for batch_idx, idx in enumerate(range(0, len(days), batch_size)):
                chunk = days[idx:idx + batch_size]
                log.info(f"Processing batch {batch_idx + 1}/{total_batches}: {chunk}")
                
                try:
                    tasks = [self.crawl_day(d, house) for d in chunk]
                    results = await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True), 
                        timeout=300
                    )
                    
                    for i, result in enumerate(results):
                        if isinstance(result, Exception):
                            log.error(f"Failed to crawl {chunk[i]}: {result}")
                        elif result:
                            self._save(result, out)
                        
                except asyncio.TimeoutError:
                    log.warning(f"Batch {chunk} timed out, continuing...")
                    continue
                except Exception as e:
                    log.error(f"Error processing batch {chunk}: {e}")
                    continue
                
                # Small delay between batches
                if batch_idx < total_batches - 1:
                    await asyncio.sleep(1)
            
            log.info(f"Crawling complete. Processed {self.processed_count} sitting days.")
            
        except Exception as e:
            log.error(f"Error in crawl method: {e}")
            raise

# ───────────────────────── CLI helpers ───────────────────────────────

def _parse_year(s: str) -> tuple[int, Optional[int], bool]:
    """Parse year argument, supporting decade literals like '1860s'."""
    if re.fullmatch(r"\d{4}s", s):  # decade literal like 1860s
        y = int(s[:-1])
        return y, y + 9, True
    return int(s), None, False

def main():
    p = argparse.ArgumentParser(
        "Hansard crawler v4.2",
        description="Crawl Historic Hansard debates from UK Parliament"
    )
    p.add_argument("start", help="start year or decade e.g. 1864 or 1860s")
    p.add_argument("end", nargs="?", help="end year (if range)")
    p.add_argument("--house", choices=["commons", "lords"], help="filter house")
    p.add_argument("--out", type=Path, default=Path("data/hansard"), help="output directory")
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="concurrent requests")
    p.add_argument("--verbose", "-v", action="store_true", help="verbose logging")
    
    args = p.parse_args()
    
    if args.verbose:
        log.setLevel(logging.DEBUG)

    try:
        s_year, _, dec_flag = _parse_year(args.start)
        e_year = int(args.end) if args.end else (s_year if not dec_flag else s_year + 9)
        
        if s_year > e_year:
            log.error("Start year cannot be after end year")
            sys.exit(1)
        
        log.info(f"Crawling Hansard data from {s_year} to {e_year}")
        if args.house:
            log.info(f"Filtering to {args.house} only")
        
        # Ensure output directory exists
        args.out.mkdir(parents=True, exist_ok=True)
        
    except ValueError as e:
        log.error(f"Invalid year format: {e}")
        sys.exit(1)

    async def _run():
        try:
            async with HansardCrawler(args.concurrency) as crawler:
                await crawler.crawl(s_year, e_year, args.out, args.house, dec_flag)
        except KeyboardInterrupt:
            log.info("Crawling interrupted by user")
        except Exception as e:
            log.error(f"Crawling failed: {e}")
            sys.exit(1)

    # Handle event loop for different environments
    try:
        if nest_asyncio and asyncio.get_event_loop().is_running():
            nest_asyncio.apply()
            asyncio.run(_run())
        else:
            asyncio.run(_run())
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            log.error("Event loop conflict. Try running in a fresh Python session.")
            sys.exit(1)
        raise

if __name__ == "__main__":
    main()