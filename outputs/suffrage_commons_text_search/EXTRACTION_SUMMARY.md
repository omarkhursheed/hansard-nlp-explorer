# Suffrage Speeches Extraction Summary

**Generated:** 2025-11-05
**Method:** Text search on speech content
**Chamber:** Commons only
**Time period:** 1900-1935
**Minimum speech length:** 50 words

---

## Extraction Results

**Total speeches found:** 2,958
**Date range:** 01 April 1901 to 31 October 1929
**Overall match rate:** 91.9% (2,735 matched to MPs)

---

## Keywords Used

Based on historical research into parliamentary language during the suffrage era (1900-1928):

### Core suffrage terms:
- women.*suffrage (women's suffrage, woman suffrage)
- female suffrage
- suffrage.*women (suffrage of women, suffrage for women)
- votes for women
- enfranchise.*women (enfranchisement of women)
- women.*enfranchise (women's enfranchisement)

### Movement terminology:
- suffragette (militant wing, WSPU, 1903+)
- suffragist (constitutional campaigners)

### Parliamentary/legal terms from 1918 Act debates:
- representation of the people (The 1918 Act)
- equal franchise (The 1928 Act)
- extension of the franchise
- women.*franchise, female.*franchise, franchise.*women
- parliamentary.*franchise.*women
- local government.*franchise.*women

### Voter terminology:
- women voters, women electors, female.*electors
- qualification.*women
- property qualification.*women (1918 Act: women 30+ with property)

### Legal/administrative:
- sex disqualification (Sex Disqualification Act)
- wspu, women.*social.*political.*union

---

## Three-Era Analysis

### Pre-1918: No Women's Suffrage (1900-1917)
- **Total speeches:** 1,644
- **Matched to MPs:** 1,523 (92.6%)
- **Gender breakdown:**
  - Male: 1,523 (100%)
  - Female: 0 (0%)

This era covers the suffrage movement's most intense campaign period, including:
- WSPU formation (1903)
- Militant suffragette tactics (1906+)
- Multiple failed suffrage bills
- WWI suspension of suffrage campaigning (1914-1917)

### 1918-1927: Partial Women's Suffrage
- **Total speeches:** 699
- **Matched to MPs:** 665 (95.1%)
- **Gender breakdown:**
  - Male: 647 (97.3%)
  - Female: 18 (2.7%)

This era covers:
- Representation of the People Act 1918 (women 30+ with property)
- Nancy Astor becomes first female MP to take seat (1919)
- Gradual increase in female MPs
- Continued campaign for equal franchise

### Post-1928: Equal Suffrage (1928-1935)
- **Total speeches:** 615
- **Matched to MPs:** 547 (88.9%)
- **Gender breakdown:**
  - Male: 509 (93.1%)
  - Female: 38 (6.9%)

This era covers:
- Equal Franchise Act 1928 (all women 21+)
- Women become electoral majority
- Increased female representation in Parliament
- Women's participation in all aspects of political life

---

## Speeches by Year

| Year | Speeches | Notes |
|------|----------|-------|
| 1900 | 38 | |
| 1901 | 21 | |
| 1902 | 33 | |
| 1903 | 15 | WSPU founded |
| 1904 | 25 | |
| 1905 | 34 | |
| 1906 | 65 | "Suffragette" term coined |
| 1907 | 39 | |
| 1908 | 104 | Peak militancy begins |
| 1909 | 82 | |
| 1910 | 118 | |
| 1911 | 96 | |
| 1912 | 153 | Height of window-smashing campaign |
| 1913 | 379 | **Peak year** - Cat and Mouse Act |
| 1914 | 61 | WWI begins, suffrage campaign suspended |
| 1915 | 13 | |
| 1916 | 45 | |
| 1917 | 323 | Representation Act debates begin |
| 1918 | 171 | Representation of the People Act passed |
| 1919 | 135 | Nancy Astor elected |
| 1920 | 83 | |
| 1921 | 24 | |
| 1922 | 57 | |
| 1923 | 26 | |
| 1924 | 58 | |
| 1925 | 46 | |
| 1926 | 26 | |
| 1927 | 73 | |
| 1928 | 143 | Equal Franchise Act passed |
| 1929 | 35 | First election with equal suffrage |
| 1930 | 38 | |
| 1931 | 156 | |
| 1932 | 23 | |
| 1933 | 51 | |
| 1934 | 31 | |
| 1935 | 138 | |

---

## Data Quality

### Match Rate by Era:
- Pre-1918: 92.6% (excellent for historical data)
- 1918-1927: 95.1% (excellent)
- Post-1928: 88.9% (good)

### Gender Coverage:
- All matched speeches have gender information
- Female MP representation increases appropriately over time:
  - 0% before 1918 (historically accurate - no female MPs)
  - 2.7% in partial suffrage era (18 speeches by women)
  - 6.9% in equal suffrage era (38 speeches by women)

---

## Files

- `speeches.parquet` - Full dataset (2,958 speeches)
- `speeches_sample.csv` - First 500 speeches for quick inspection

---

## Usage Examples

### Load the data:
```python
import pandas as pd

speeches = pd.read_parquet('outputs/suffrage_commons_text_search/speeches.parquet')
```

### Filter by era:
```python
pre_1918 = speeches[speeches['year'] < 1918]
partial = speeches[(speeches['year'] >= 1918) & (speeches['year'] < 1928)]
post_1928 = speeches[speeches['year'] >= 1928]
```

### Get female MP speeches:
```python
female_speeches = speeches[
    (speeches['matched_mp'] == True) &
    (speeches['gender'] == 'F')
]
```

### Analyze by year:
```python
by_year = speeches.groupby(['year', 'gender']).size().unstack(fill_value=0)
```

---

## Historical Context

### Key Legislation:
- **1918 Representation of the People Act**: Gave vote to women 30+ with property (8.5M women)
- **1928 Equal Franchise Act**: Extended vote to all women 21+ (5M more women)

### Key Figures:
- **Nancy Astor**: First female MP to take seat (1919, Conservative)
- **Emmeline Pankhurst**: WSPU leader, militant suffragette
- **Millicent Fawcett**: NUWSS leader, constitutional suffragist

### Parliamentary Language Evolution:
- Early period (1900-1910): Focus on "women's suffrage," "female franchise"
- Militant period (1906-1914): "Suffragette," "votes for women"
- Legislative period (1917-1928): "Representation of the people," "extension of the franchise," "equal franchise"

---

## Extraction Method

The extraction used regex pattern matching on speech text to find speeches mentioning women's suffrage. This approach captures:

1. **Direct debate speeches**: Speeches in debates explicitly about suffrage bills
2. **Tangential mentions**: Speeches on other topics that reference suffrage
3. **Opposition speeches**: Arguments against suffrage (important for balanced analysis)
4. **Implementation discussions**: Debates about how suffrage would work in practice

This is more comprehensive than title-only search, which only captures explicit suffrage debates.

---

## Next Steps

Potential analyses with this dataset:

1. **Language evolution**: How did suffrage discourse change across the three eras?
2. **Gender differences**: Did female MPs use different language when discussing suffrage?
3. **Party differences**: How did Conservative vs Liberal vs Labour MPs differ in their suffrage speeches?
4. **Opposition arguments**: What arguments were used against suffrage, and how did they evolve?
5. **Implementation focus**: After 1918, how did debate shift from "whether" to "how"?
