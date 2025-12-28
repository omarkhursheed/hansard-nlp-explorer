# 🏛️ House Members Pipeline

This pipeline reconstructs and gender-tags all members of the **UK House of Commons** and **House of Lords** between **1803–2005** by combining data from multiple open parliamentary sources.  

It merges and reconciles person records from:
- **ParlParse (PP)** — The official Hansard data source.
- **EveryPolitician (EP)** — Popolo-formatted metadata and Wikidata links.
- **Historic Hansard (HH)** — Archived Hansard website covering 1803–2005.

The result is a comprehensive, deduplicated, and gender-annotated dataset of all members of both chambers during the Historic Hansard period.

---

## 📂 Overview of the Pipeline

### 1. Data Acquisition

1. **ParlParse (PP):**
   - Clone or download the Popolo file `people.json` file from the official ParlParse repository.
   - File location:
     ```bash
     src/hansard/data/hansard_raw/ParlParse_git/people.json
     ```

2. **EveryPolitician (EP):**
   - Download the Popolo file (`people.json`) from the EveryPolitician GitHub repository.
   - Rename to `ep_people.json` and place it at:
     ```bash
     src/hansard/data/hansard_raw/Everypolitician_git/ep_people.json
     ```

3. **Historic Hansard (HH):**
   - Run the crawler to fetch all member pages and metadata:
     ```bash
     python fetch_speakers_historic_hansard.py
     ```
   - This fetches and parses all entries from  
     [https://api.parliament.uk/historic-hansard/people/](https://api.parliament.uk/historic-hansard/people/)  
     and saves them as a structured parquet.

---

### 2. Parsing Source Databases

Each source is parsed and normalized into consistent metadata schemas covering:
- Person identifiers  
- Chamber membership  
- Honorifics  
- Birth/death years  
- Wikidata/Parliament URLs  
- Membership start and end dates  

Scripts:

| Script | Description |
| ------- | ------------ |
| `get_PP_hansard_members.py` | Parses ParlParse people.json to extract Commons/Lords members and metadata between 1803–2005. |
| `get_EP_hansard_members.py` | Parses EveryPolitician ep_people.json in the same manner. |

Each script outputs a processed parquet file under  
`src/hansard/data/processed_fixed/metadata/house_members/`.

---

### 3. Merging the Databases

This step integrates the three datasets (PP, EP, HH) into a single master file.

#### Step 3.1 — Merge PP and EP

- PP is treated as **ground truth**.  
- Merge type: **Left join on PP**.  
- Extra metadata from EP (Wikidata IDs, Wikipedia links, etc.) is added for overlapping members.  
- Script:
  ```bash
  python merge_PP_EP_memberdata.py

#### Step 3.2 — Merge with HH

- The PP–EP merged dataset is then joined with HH using:
  - Exact IDs  
  - Name matches  
  - Fuzzy name similarity  

- HH metadata (titles, lifespans, etc.) is transferred to matching records.  
- Unmatched HH members (rare, obscure historical names) are appended.  
- Script:
  ```bash
  python merge_PP_HH_members.py

#### Step 3.2 — Merge with HH

- The PP–EP merged dataset is then joined with HH using:
  - Exact IDs  
  - Name matches  
  - Fuzzy name similarity  

- HH metadata (titles, lifespans, etc.) is transferred to matching records.  
- Unmatched HH members (rare, obscure historical names) are appended.  
- Script:
  ```bash
  python merge_PP_HH_members.py
  ```

#### Step 3.3 — Final Merge Output

- The merged dataset includes:
  - `person_id`: primary identifier across all sources  
  - `aliases_norm`: normalized variants of names (for matching consistency)  
  - `id_historichansard_url`: identifies extra HH-only entries  
  - Multiple rows per member to represent distinct parliamentary stints  

---

### 4. Gender Tagging Pipeline

Each unique `person_id` is assigned a gender (`M` or `F`) with traceable provenance recorded in `gender_source`.

Script:
```bash
python infer_gender_house_members.py
```

The pipeline proceeds through the following inference hierarchy:

| Step | Source | Logic | Output columns |
|------|---------|--------|----------------|
| **1. EveryPolitician** | `gender` field (if available) | Assign `M` / `F` | `gender_source = everypolitician` |
| **2. Historic Hansard honorifics** | Title prefixes (`Mr`, `Miss`, `Lord`, etc.) | Rule-based match | `gender_source = HH honorific` |
| **3. ParlParse honorifics** | Extracted from `other_names` or `honorific_prefix` | Rule-based match | `gender_source = PP honorific` |
| **4. Wikidata enrichment** | Fetch `P21` via SPARQL (using `fetch_member_gender_wikidata.py`) | API lookup | `gender_source = wikidata_P21` |
| **5. Historical rules** | Assign males to pre-suffrage eras | Commons <1918, Lords <1958 → `M` | `gender_source = rule_preXXXX` |
| **6. Curated female lists** | Fetch via `fetch_female_members_wikidata.py` | Match names in official Wikipedia lists of female MPs and Lords | `gender_source = wikidata_curated_*` |
| **7. Name-based inference** | Uses `gender-guesser` library | First-name model (~97–98% accuracy) | `gender_source = fname_inference` |
| **8. Optional fallback** | Default unknowns to `M` (commented out by default) | Conservative bias correction | Not applied |

Each step logs the number of new gender assignments incrementally.

---

### 5. Final Output

The final enriched dataset:
```bash
src/hansard/data/processed_fixed/metadata/house_members/house_members_gendered.parquet
```

**Columns include:**
- `person_id` — Unique identifier (primary key)  
- `person_name` — Canonical name  
- `organization_id` — Chamber (`house-of-commons`, `house-of-lords`)  
- `membership_start_date` / `membership_end_date`  
- `id_wikidata`, `id_parliamentdotuk`, etc.  
- `honorific_prefix`  
- `aliases_norm`  
- `gender_inferred` — “M” / “F” / None  
- `gender_source` — Provenance label  

---

## ⚙️ Environment Setup

```bash
# Recommended packages
pip install pandas numpy requests beautifulsoup4 SPARQLWrapper gender-guesser lxml rapidfuzz

# Optional: enable Jupyter/Polars for exploration
pip install jupyter polars
```

---

## 🧭 Execution Order Summary

```bash
# 1. Fetch source data
python fetch_speakers_historic_hansard.py

# 2. Parse PP and EP datasets
python get_PP_hansard_members.py
python get_EP_hansard_members.py

# 3. Merge all databases
python merge_PP_EP_memberdata.py
python merge_PP_HH_members.py

# 4. Infer and assign gender
python house_members_gender_inference.py
```

---

## 📊 Coverage Summary

| Dataset | Members Covered | Description |
| Total Members | 14,752 unique members (person_ids+id_historichansard_url) | Unified dataset; Tagged M/F + provenance; ~64 unknown gender | 

---

## 🧩 Directory Structure

```bash
src/hansard/data/
├── hansard_raw/
│   ├── ParlParse_git/people.json
│   └── Everypolitician_git/ep_people.json
├── processed_fixed/
│   └── metadata/house_members/
│       ├── PP_HH_members_combined.parquet
│       ├── house_members_gendered.parquet
│       ├── EP_commons_members.parquet
│       ├── historic_hansard_speakers.parquet
│       ├── PP_house_members_1803_2005.parquet
│       ├── PP_EP_house_members_combined.parquet
│       └── ...
└── word_lists/
    └── gendered_honorifics.json
```

---

## 🪪 Attribution

- **ParlParse**: © UK Parliament, open data under [OGL v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).  
- **EveryPolitician**: © mySociety, under the same OGL.  
- **Historic Hansard Archive**: Public record via Parliament API.  
- **Wikidata/Wikipedia**: Data licensed under CC-BY-SA 3.0.

---

## 🧠 Maintainer Notes
 
- The gender inference pipeline is deterministic — reruns yield identical output.  
- Honorific lists (`gendered_honorifics.json`) can be extended manually if needed.  

---
## 📜 License

**Last updated:** October 2025  
```
