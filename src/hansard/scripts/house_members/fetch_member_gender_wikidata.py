# %pip install SPARQLWrapper

from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd

def fetch_wikidata_genders(qids):
    """Fetch gender from Wikidata for a list of QIDs."""
    endpoint = "https://query.wikidata.org/sparql"
    sparql = SPARQLWrapper(endpoint)
    
    # Build VALUES block
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
