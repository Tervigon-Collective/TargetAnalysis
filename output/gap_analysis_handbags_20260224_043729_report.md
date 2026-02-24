# Gap Analysis Report

Generated: 2026-02-24T04:37:29.780909
Input: ChromaDB
Products: 300
Outputs: gap_analysis_handbags_20260224_043729.csv, gap_analysis_handbags_20260224_043729.json

---

## 1. Coverage gap in core professional price band

Coverage gap: Universal Thread has 16 SKU gap vs A New Day in $0-20 other (33 vs 17).

**Why this matters:**
- Mid-tier ($60–90) = sweet spot for AOV
- Work totes = office return + higher basket
- SKU gap = directly actionable buy decision

**Top coverage gaps:**

- $0-20 other: Universal Thread vs A New Day — 16 SKU gap (33 vs 17)
- $20-40 other: Universal Thread vs A New Day — 13 SKU gap (40 vs 27)
- $40-60 totes: Universal Thread vs A New Day — 5 SKU gap (5 vs 0)
- $20-40 totes: A New Day vs Universal Thread — 1 SKU gap (8 vs 7)

---

## 2. Visual evidence


UMAP scatter: (not generated)

---

## 3. Summary

- **Products:** 300
- **Clustering:** text embeddings (CLIP/TF-IDF)
- **Style clusters (K-Means k=5):** 5
- **Price bands:** 5 (0-20, 20-40, 40-70, 70-120, 120+ + unknown)

---

## 4. Brand comparison: Universal Thread, A New Day

| Metric | Universal Thread | A New Day |
|--------|--------|--------|
| **SKU count** | 57 | 90 |
| **Avg price** | 24.2 | 22.7 |
| **Avg rating** | 4.40 | 4.38 |
| **Sale %** | 9% | 17% |
| **Best seller %** | 0% | 0% |
| **New arrival %** | 0% | 0% |
| **In-stock %** | 0% | 0% |

**Price band distribution:**

- 0-20: Universal Thread=18, A New Day=34
- 20-40: Universal Thread=36, A New Day=48
- 40-70: Universal Thread=3, A New Day=8
- 70-120: Universal Thread=0, A New Day=0
- 120+: Universal Thread=0, A New Day=0
- unknown: Universal Thread=0, A New Day=0

**Cluster distribution:**

- Cluster 0: Universal Thread=22, A New Day=36
- Cluster 1: Universal Thread=28, A New Day=37
- Cluster 2: Universal Thread=7, A New Day=15
- Cluster 3: Universal Thread=0, A New Day=0
- Cluster 4: Universal Thread=0, A New Day=2

---

## 5. Style clusters (SKU count, avg price, avg rating, dominant brand)

- **Cluster 0:** SKUs=73, avg_price=23.8, avg_rating=4.53, top_brand=A New Day
- **Cluster 1:** SKUs=92, avg_price=124.6, avg_rating=4.32, top_brand=A New Day
- **Cluster 2:** SKUs=52, avg_price=19.1, avg_rating=4.41, top_brand=Gap
- **Cluster 3:** SKUs=19, avg_price=11.1, avg_rating=4.94, top_brand=Gap
- **Cluster 4:** SKUs=64, avg_price=633.0, avg_rating=4.50, top_brand=Gap

---

## 6. SKU density (cluster × price band)

```
price_band      0-20  120+  20-40  40-70  70-120  All
cluster_kmeans                                       
0                 27     0     42      3       1   73
1                 18     2     56     14       2   92
2                 24     0     25      3       0   52
3                 19     0      0      0       0   19
4                  7     9     27     13       8   64
All               95    11    150     33      11  300
```

*(Empty or near-zero cells = price–style gap)*

---

## 7. Rating × review quadrant

- **Proven winner:** 142
- **Weak:** 107
- **Hidden opportunity:** 42
- **Market mismatch:** 9

Hidden winners (high rating, low reviews) by cluster:

- Cluster 0: 8
- Cluster 1: 25
- Cluster 2: 7
- Cluster 4: 2

---

## 8. Brand dominance (per cluster)

- **Cluster 0:** top brand share = 49% — mixed/white-space
- **Cluster 1:** top brand share = 40% — mixed/white-space
- **Cluster 2:** top brand share = 50% — mixed/white-space
- **Cluster 3:** top brand share = 100% — dominance
- **Cluster 4:** top brand share = 78% — dominance

---

## 9. Color diversity (by cluster)

- **Cluster 0:** avg color_count=1.4, main families: {'black': 58, 'unknown': 5, 'bold': 5}
- **Cluster 1:** avg color_count=1.4, main families: {'neutral': 47, 'unknown': 21, 'bold': 14}
- **Cluster 2:** avg color_count=3.4, main families: {'unknown': 19, 'bold': 15, 'black': 9}
- **Cluster 3:** avg color_count=11.2, main families: {'bold': 13, 'neutral': 3, 'black': 3}
- **Cluster 4:** avg color_count=1.8, main families: {'unknown': 24, 'pastel': 15, 'neutral': 12}

---

## 10. Title keyword co-occurrence (gaps)

Top pairs (by count):
- tote + mini: 5
- crossbody + mini: 5
- tote + crossbody: 4
- clutch + crossbody: 3
- backpack + laptop: 3

Gaps (count = 0):
- tote + clutch
- tote + backpack
- tote + convertible
- tote + laptop
- tote + organizer
- clutch + backpack
- clutch + convertible
- clutch + quilted
- clutch + faux leather
- clutch + laptop
- clutch + organizer
- clutch + mini
- clutch + drawstring
- backpack + crossbody
- backpack + faux leather
- backpack + organizer
- backpack + drawstring
- crossbody + convertible
- crossbody + laptop
- crossbody + organizer
- ... and 18 more

---

## 11. Six gap analyses

### 11.1 Rating-weighted cluster strength

```
                cluster_score  sku_count
cluster_kmeans                          
0                   16.090343         73
3                   15.207963         19
1                   11.140340         92
4                    5.977060         64
2                    2.797238         52
```

- High score + low SKU count → expand. Low score + high SKU count → over-indexed.

### 11.2 New vs established

```
cluster_kmeans
0    0
1    0
2    0
3    0
4    0
```

- If new SKUs concentrated in one cluster → not innovating in other styles.

### 11.3 Bestseller gap

No best_seller=True in dataset.

### 11.4 Clearance pattern

```
cluster_kmeans
0    0.0
1    0.0
2    0.0
3    0.0
4    0.0
```

- High clearance % in a cluster → possible weak demand.

### 11.5 In-stock pressure

```
cluster_kmeans
0     94.520548
1     91.304348
2    100.000000
3      0.000000
4     75.000000
```

- High out-of-stock % → demand signal.

---

*Internal trend map; competitor comparison when data available.*
