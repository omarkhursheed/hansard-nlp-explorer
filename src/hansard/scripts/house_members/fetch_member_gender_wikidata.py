"""
Wikidata Gender Fetcher via SPARQL

This tiny utility queries the public Wikidata SPARQL endpoint to retrieve
gender (P21) for a list of Wikidata entity IDs (QIDs). It returns a dict
mapping QID -> "M" / "F" / None (if gender is missing/other/unknown).

Notes:
- Expects QIDs like "Q8016" (no URI prefix).
- Skips NaN/None entries.
- For very large lists, consider chunking to avoid long queries and to be
  polite to the endpoint’s rate limits.
"""

# %pip install SPARQLWrapper

from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd

def fetch_wikidata_genders(qids):
    """
    Query Wikidata for gender (P21) labels for the given QIDs.

    Args:
        qids (Iterable[str|None]): An iterable of Wikidata entity IDs (e.g., ["Q8016", "Q42"]).
                                   Any null/NaN values are ignored.

    Returns:
        dict[str, str|None]: A mapping from QID to a short code:
            - "F" for female
            - "M" for male
            - None if gender is missing/other/unknown

    Behavior:
        - Sends a single SPARQL query with a VALUES block listing all QIDs.
        - Uses English labels via the wikibase:label service.
        - Interprets the returned gender label case-insensitively.

    Caveats:
        - The Wikidata public endpoint enforces rate limits and may throttle
          very large VALUES queries. If you see timeouts, split `qids` into
          smaller batches (e.g., 100–500 QIDs per call).
        - Non-binary genders will map to None by default here; extend the mapping
          if you need finer-grained categories.
    """
    endpoint = "https://query.wikidata.org/sparql"
    sparql = SPARQLWrapper(endpoint)

    # Build VALUES block (skip nulls/NaNs)
    values = " ".join(f"wd:{qid}" for qid in qids if pd.notna(qid))
    query = f"""
    SELECT ?item ?genderLabel WHERE {{
      VALUES ?item {{ {values} }}
      ?item wdt:P21 ?gender .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    # Map QID → gender short code
    mapping = {}
    for r in results["results"]["bindings"]:
        qid = r["item"]["value"].split("/")[-1]
        gender = r["genderLabel"]["value"].lower()
        if "female" in gender:
            mapping[qid] = "F"
        elif "male" in gender:
            mapping[qid] = "M"
        else:
            mapping[qid] = None
    return mapping
