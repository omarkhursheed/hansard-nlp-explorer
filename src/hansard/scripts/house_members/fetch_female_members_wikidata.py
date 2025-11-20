"""
Female Parliamentarians Utilities

This module provides two helpers:

1) fetch_female_mps()
   Queries the public Wikidata SPARQL endpoint for female Members of the
   UK House of Commons (P21 = female; P39 is-a Commons MP). Returns a
   tidy DataFrame of QIDs and labels.

2) scrape_female_lords()
   Scrapes Wikipedia’s “List of female members of the House of Lords”
   page and returns a DataFrame of names parsed from the wikitables.

Notes
-----
- Be considerate of Wikidata/Wikipedia rate limits; cache or batch if needed.
- The Wikipedia table structure occasionally changes; parsing is best-effort.
"""

from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd
import requests
from bs4 import BeautifulSoup


def fetch_female_mps() -> pd.DataFrame:
    """
    Query Wikidata for female Members of the UK House of Commons.

    The query selects people (?person) with:
      - sex or gender (P21) = female (Q6581072)
      - a position held (P39) statement whose value is a subclass-of
        (P279*) "Member of the House of Commons of the United Kingdom" (Q16707842).
    Optional qualifiers P580/P582 (start/end) are included only to enable
    ordering by start date in case of multiple stints.

    Returns
    -------
    pandas.DataFrame
        Columns:
          - id_wikidata: str (QID, e.g., 'Q42')
          - name: str (English label from Wikidata)

    Caveats
    -------
    - The public SPARQL endpoint has rate limits; for large-scale use, batch or cache.
    - This returns unique (personLabel, QID) pairs present in the results; it
      does not deduplicate across multiple P39 statements.
    """
    query = """
    SELECT ?person ?personLabel ?start ?end WHERE {
      ?person wdt:P21 wd:Q6581072; p:P39 ?statement .
      ?statement ps:P39 ?position .
      ?position wdt:P279* wd:Q16707842 .  # Commons MPs
      OPTIONAL { ?statement pq:P580 ?start. }
      OPTIONAL { ?statement pq:P582 ?end. }
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }
    ORDER BY ?start
    """
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    rows = []
    for r in results["results"]["bindings"]:
        qid = r["person"]["value"].split("/")[-1]
        rows.append({
            "id_wikidata": qid,
            "name": r["personLabel"]["value"],
        })
    return pd.DataFrame(rows)


def scrape_female_lords() -> pd.DataFrame:
    """
    Scrape the Wikipedia page for female members of the House of Lords.

    Strategy
    --------
    - Download the page:
        https://en.wikipedia.org/wiki/List_of_female_members_of_the_House_of_Lords
    - Identify tables with class 'wikitable' whose headers include both
      "name" and "year joined".
    - For each qualifying table, read the third column as the name cell,
      strip markup, and normalize to a simple display name (drop text after commas).

    Returns
    -------
    pandas.DataFrame
        Columns:
          - raw_name: str

    Caveats
    -------
    - Wikipedia layout can change; header detection is heuristic.
    - Some rows may include footnotes or unusual formatting; names are
      extracted best-effort and lightly normalized.
    """
    url = "https://en.wikipedia.org/wiki/List_of_female_members_of_the_House_of_Lords"
    html = requests.get(url).text
    soup = BeautifulSoup(html, "html.parser")

    lords_names = []
    for table in soup.find_all("table", {"class": "wikitable"}):
        headers = [h.get_text(" ", strip=True).lower() for h in table.find_all("th")]
        if not ("name" in " ".join(headers) and "year joined" in " ".join(headers)):
            continue
        for row in table.find_all("tr")[1:]:
            cols = row.find_all("td")
            if len(cols) < 3:
                continue
            raw_name = cols[2].get_text(" ", strip=True)
            if raw_name and raw_name.lower() != "name":
                clean = raw_name.split(",")[0].strip()
                lords_names.append(clean)

    df_lords = pd.DataFrame({"raw_name": lords_names})
    return df_lords
