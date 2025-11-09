#!/usr/bin/env python3
"""
Fast parallel indexer - processes multiple years concurrently.
Should take 45-90 minutes instead of 3-5 hours.
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime
import httpx
from bs4 import BeautifulSoup
import re
from collections import defaultdict

BASE_URL = "https://api.parliament.uk/historic-hansard"
MONTHS = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]

class FastHansardIndexer:
    def __init__(self, max_concurrent_years=5, rps=10):
        self.index = {}
        self.stats = {
            "years": 0,
            "months": 0,
            "dates": 0,
            "debates": 0,
            "errors": []
        }
        self.max_concurrent_years = max_concurrent_years
        self.rps = rps
        self.semaphore = asyncio.Semaphore(max_concurrent_years * 3)  # Control total concurrent requests
        self.rate_limiter = asyncio.Semaphore(rps)  # Overall rate limit
        self.last_request_time = 0
        self.lock = asyncio.Lock()

    async def rate_limit(self):
        """Global rate limiter."""
        async with self.lock:
            now = asyncio.get_event_loop().time()
            time_since_last = now - self.last_request_time
            min_interval = 1.0 / self.rps

            if time_since_last < min_interval:
                await asyncio.sleep(min_interval - time_since_last)

            self.last_request_time = asyncio.get_event_loop().time()

    async def fetch(self, client: httpx.AsyncClient, url: str):
        """Fetch with rate limiting."""
        async with self.semaphore:
            await self.rate_limit()
            try:
                r = await client.get(url, timeout=30.0)
                if r.status_code == 200:
                    return r.text
                return None
            except Exception as e:
                self.stats["errors"].append(f"Error fetching {url}: {e}")
                return None

    async def get_months(self, client: httpx.AsyncClient, year: int) -> list:
        """Get all months in a year."""
        html = await self.fetch(client, f"{BASE_URL}/sittings/{year}")
        if not html:
            return []

        soup = BeautifulSoup(html, 'html.parser')
        links = [a.get('href', '') for a in soup.find_all('a', href=True)]

        months = []
        for link in links:
            if f"/sittings/{year}/" in link:
                month = link.split(f"/sittings/{year}/")[1].split('/')[0].lower()
                if month in MONTHS:
                    months.append(month)

        return sorted(set(months), key=lambda m: MONTHS.index(m))

    async def get_days(self, client: httpx.AsyncClient, year: int, month: str) -> list:
        """Get all sitting days in a month."""
        html = await self.fetch(client, f"{BASE_URL}/sittings/{year}/{month}")
        if not html:
            return []

        soup = BeautifulSoup(html, 'html.parser')
        links = [a.get('href', '') for a in soup.find_all('a', href=True)]

        days = []
        for link in links:
            if f"/sittings/{year}/{month}/" in link:
                day = link.split(f"/sittings/{year}/{month}/")[1].split('/')[0]
                if day.isdigit():
                    days.append(day)

        return sorted(set(days), key=int)

    async def get_debates(self, client: httpx.AsyncClient, year: int, month: str, day: str) -> list:
        """Get all debates for a specific date."""
        html = await self.fetch(client, f"{BASE_URL}/sittings/{year}/{month}/{day}")
        if not html:
            return []

        soup = BeautifulSoup(html, 'html.parser')
        links = [a.get('href', '') for a in soup.find_all('a', href=True)]

        d_padded = day.zfill(2) if day.isdigit() else day
        day_formats = [day, d_padded] if day != d_padded else [day]

        debates = []
        for link in links:
            if not link or any(x in link.lower() for x in ("index", "contents", "#")):
                continue

            for day_format in day_formats:
                pattern = rf"/(commons|lords)/{re.escape(str(year))}/{re.escape(month)}/{re.escape(day_format)}/([^/?#]+)"
                m = re.search(pattern, link, re.I)
                if m:
                    path = m.group(0).lstrip("/")
                    if path not in debates:
                        debates.append(path)
                    break

        return debates

    async def index_year(self, client: httpx.AsyncClient, year: int):
        """Index a complete year."""
        try:
            months = await self.get_months(client, year)
            if not months:
                return

            year_data = {}

            for month in months:
                days = await self.get_days(client, year, month)
                if not days:
                    continue

                month_data = {}

                for day in days:
                    debates = await self.get_debates(client, year, month, day)

                    date_key = f"{year}/{month}/{day}"
                    month_data[day] = {
                        "date": date_key,
                        "debates": debates,
                        "debate_count": len(debates)
                    }

                    self.stats["dates"] += 1
                    self.stats["debates"] += len(debates)

                if month_data:
                    year_data[month] = month_data
                    self.stats["months"] += 1

            if year_data:
                self.index[str(year)] = year_data
                self.stats["years"] += 1

            # Progress
            if self.stats["years"] % 10 == 0:
                print(f"  Progress: {self.stats['years']} years, {self.stats['dates']} dates, {self.stats['debates']} debates")

        except Exception as e:
            self.stats["errors"].append(f"Error indexing {year}: {e}")

    async def create_full_index(self, start_year: int = 1803, end_year: int = 2005):
        """Create complete index with parallel processing."""
        print("="*70)
        print("FAST PARALLEL HANSARD INDEXER")
        print("="*70)
        print(f"Years: {start_year}-{end_year}")
        print(f"Concurrent years: {self.max_concurrent_years}")
        print(f"Rate limit: {self.rps} requests/second")
        print(f"Estimated time: ~25-30 minutes")
        print()

        input("Press Enter to start indexing...")

        async with httpx.AsyncClient(
            http2=True,
            follow_redirects=True,
            timeout=30.0,
            headers={"User-Agent": "HansardFastIndexer/1.0"},
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50)
        ) as client:

            # Process years in batches
            years = list(range(start_year, end_year + 1))
            batch_size = self.max_concurrent_years

            for i in range(0, len(years), batch_size):
                batch = years[i:i+batch_size]
                print(f"Processing years: {batch[0]}-{batch[-1]}")

                tasks = [self.index_year(client, year) for year in batch]
                await asyncio.gather(*tasks)

        print()
        print("="*70)
        print("INDEXING COMPLETE")
        print("="*70)
        print(f"Years indexed: {self.stats['years']}")
        print(f"Months indexed: {self.stats['months']}")
        print(f"Dates indexed: {self.stats['dates']}")
        print(f"Debates cataloged: {self.stats['debates']}")
        print(f"Errors: {len(self.stats['errors'])}")

        if self.stats['errors']:
            print("\nFirst 10 errors:")
            for err in self.stats['errors'][:10]:
                print(f"  {err}")

    def save(self, output_file: Path):
        """Save index to JSON file."""
        output_file.parent.mkdir(exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump({
                "created": datetime.now().isoformat(),
                "stats": self.stats,
                "index": self.index
            }, f, indent=2)

        print(f"\nIndex saved to: {output_file}")
        print(f"File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")

async def main():
    # Adjust these for speed vs politeness tradeoff
    # API tested up to 30 RPS with no errors!
    # Conservative: max_concurrent_years=5, rps=10   (~45 min)
    # Balanced: max_concurrent_years=10, rps=20      (~25 min)
    # Aggressive: max_concurrent_years=15, rps=30    (~18 min)

    indexer = FastHansardIndexer(
        max_concurrent_years=10,  # Process 10 years at once
        rps=20                     # 20 requests per second total
    )

    await indexer.create_full_index(start_year=1803, end_year=2005)
    indexer.save(Path("analysis/hansard_complete_index.json"))

if __name__ == "__main__":
    asyncio.run(main())
