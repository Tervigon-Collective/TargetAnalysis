# White Space Analysis Report

## Summary

- **Focus brand:** shade & shore
- **Competitor brands:** j.crew
- **Total gap cells:** 30
- **Gap cells with competitor_count >= 2:** 21
- **Top opportunity (gap):** Bottoms | value_mass | 50_75 (gap=100.0%)

## Visualizations

- `white_space_top_opportunities.png` – Top cells by gap (specific category|mission|price cells)
- `white_space_gap_heatmap.png` – Gap by mission × price band (aggregated, no category)
- `white_space_gap_by_price_band.png` – Mean gap by price band (price lens only)
- `white_space_price_band_by_brand.png` – Price band mix by brand (% of SKUs per band)
- `white_space_category_by_brand.png` – Category coverage by brand (top categories)
- `white_space_sku_by_mission_brand.png` – Mission share (%) by brand (% of each brand's SKUs per mission)
- `white_space_mission_sku_count_heatmap.png` – Mission density (%) by brand; box labels = SKU count (reference)
- `white_space_target_vs_competitor.png` – shade & shore vs j.crew by mission

## Understanding the Charts

**Why does shade & shore vs Competitors sometimes show shade & shore at 0%?**
When the focus brand has no (or few) products in mission cells where competitor brands invest, focus_density is 0%.
Those blue bars with "0%" labels indicate white space: competitors are present, focus brand is absent = opportunity.

**Density** = % of that brand's category SKUs in that (mission, price_band) cell.
**Gap** = competitor_median_density - focus_density. Positive gap = shade & shore under-invested vs competitor brands.

## Decision Guidance

1. **High gap + high competitor_count** → Strong opportunity; consider sampling concepts.
2. **High gap + low competitor_count** → Monitor; may be niche.
3. **Low gap (shade & shore ahead)** → Maintain or optimize; no expansion needed.