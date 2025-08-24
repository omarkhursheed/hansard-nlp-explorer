from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd
import requests
from bs4 import BeautifulSoup

# --- Fetch female MPs from Wikidata ---
def fetch_female_mps():
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


# --- Scrape Lords ---
def scrape_female_lords():
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
