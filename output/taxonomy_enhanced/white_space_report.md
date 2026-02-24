# White Space Analysis Report

## Summary

- **Focus brand:** a new day
- **Competitor brands:** abercrombie
- **Total gap cells:** 13
- **Gap cells with competitor_count >= 2:** 13
- **Top opportunity (gap):** Abercrombie x Kemo Sabe | statement_fashion | 35_50 (gap=100.0%)

## Visualizations

- `white_space_top_opportunities.png` – Top cells by gap (specific category|mission|price cells)
- `white_space_gap_heatmap.png` – Gap by mission × price band (aggregated, no category)
- `white_space_gap_by_price_band.png` – Mean gap by price band (price lens only)
- `white_space_price_band_by_brand.png` – Price band mix by brand (% of SKUs per band)
- `white_space_category_by_brand.png` – Category coverage by brand (top categories)
- `white_space_sku_by_mission_brand.png` – Mission share (%) by brand (% of each brand's SKUs per mission)
- `white_space_target_vs_competitor.png` – a new day vs abercrombie by mission

## Understanding the Charts

**Why does a new day vs Competitors sometimes show a new day at 0%?**
When the focus brand has no (or few) products in mission cells where competitor brands invest, focus_density is 0%.
Those blue bars with "0%" labels indicate white space: competitors are present, focus brand is absent = opportunity.

**Density** = % of that brand's category SKUs in that (mission, price_band) cell.
**Gap** = competitor_median_density - focus_density. Positive gap = a new day under-invested vs competitor brands.

## Decision Guidance

1. **High gap + high competitor_count** → Strong opportunity; consider sampling concepts.
2. **High gap + low competitor_count** → Monitor; may be niche.
3. **Low gap (a new day ahead)** → Maintain or optimize; no expansion needed.