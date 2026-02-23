# Enhanced Taxonomy Report

**Date:** 20260223_185444
**Input:** products_mission_enriched.csv

## Clustering quality

- **Mean silhouette (per silhouette):** 0.0245
- **Mean Davies-Bouldin:** 1.92

| Silhouette | N | Clusters | Silhouette | Davies-Bouldin |
|------------|---|----------|------------|----------------|
| clutch | 14 | 2 | 0.061 | 2.46 |
| tote | 31 | 6 | -0.011 | 0.95 |
| accessory | 35 | 7 | 0.033 | 1.76 |
| shoulder | 25 | 5 | 0.021 | 2.55 |
| crossbody | 29 | 5 | -0.019 | 1.62 |
| satchel | 10 | 2 | 0.063 | 2.16 |

## Comparison (Target vs Gap)

| Metric | Target | Gap |
|--------|--------|-----|
| Count | 173 | 0 |
| Median price | $25.0 | $nan |

## Upgrades applied

- Brand tokens from CSV brand column (source=retailer, brand=Champion/Kipling/etc.)
- Statistical stopwords (configurable doc-freq threshold)
- Hybrid embeddings: TF-IDF + bigrams + price bucket + material + is_sale
- HDBSCAN within each silhouette (auto cluster count, noise → nearest cluster)
- Structural word removal (zip, strap, pockets, closure)
- NEW (viz only): PCA/UMAP/t-SNE 2D plots + per-silhouette subcluster plots (reduces clutter)

## Segments (sample)

- **accessory – unknown | new day charms / day charms straps | E** (17 items)
- **accessory – single** (5 items)
- **accessory – unknown | straps acrylic / charms straps acrylic** (13 items)
- **tote – unknown | universal thread / interior exterior | Mid** (26 items)
- **tote – single** (5 items)
- **crossbody – unknown | universal thread / interior exterior |** (23 items)
- **crossbody – single** (3 items)
- **crossbody – leather | universal thread / double gusset | Ent** (3 items)
- **shoulder – canvas_cotton | slouchy shoulder / cotton polyest** (3 items)
- **shoulder – recycled | recycled polyester / 100 recycled poly** (5 items)
- **shoulder – unknown | faux suede / tan shoulder | Mid** (5 items)
- **clutch – unknown | kiss lock / credit card | Entry** (4 items)
- **clutch – unknown | minaudiere clutch / interior exterior | M** (10 items)