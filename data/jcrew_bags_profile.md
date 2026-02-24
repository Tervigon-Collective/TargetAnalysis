# J.Crew Bags Products – Data Profile & Normalization Guide

## Column Overview

| Column | Type | Sample | Notes |
|--------|------|--------|-------|
| availability_text | str | (empty) | Usually empty in J.Crew data |
| average_rating | float | 4.83, 4.0 | Decimal ratings |
| brand | str | J.Crew | Consistent |
| care_instructions | str | "Careers", long return policy text | **Junk**: "Careers" is wrong element; sometimes return policy |
| category_breadcrumb | str | "Home > women > bags" | Use last segment for leaf_category |
| color_options | JSON array | `["Blonde Espresso", "Black"]` | **Normalize**: Convert to pipe-separated |
| description | str | Long product copy | **Junk**: Row 10 has JavaScript blob (Boomerang script) |
| dimensions | str | "10\"H x 15\"W x 3 1/4\"D" | Often empty for non-bags |
| image_count | int | 11 | Integer |
| image_url | str | jcrew.com/s7-img-facade/... | Valid URLs |
| images_list | JSON array | `["url1", "url2"]` | **Normalize**: Convert to pipe-separated |
| is_sale | bool str | "True", "False" | Should validate against price_original |
| material_text | str | "100% leather. ..." | Generally good |
| name | str | Product title | Maps to product_name |
| price_currency | str | USD | Consistent |
| price_current | float | 274.19 | Primary price |
| price_original | float | 178 | When = price_current → clear (no sale) |
| product_id | str | CM933, CR297_BR8760 | Item code; sku includes color variant |
| product_url | str | jcrew.com/p/... | Valid URLs |
| promo_excluded | str | (empty) | |
| rating_count | int | 151 | Often same as review_count |
| review_count | int | 151 | |
| scrape_date | str | 2026-02-23T12:56:40Z | ISO timestamp |
| sku | str | CM933_EE7690 | product_id + color code |
| sold_shipped_by | str | J.Crew | Usually "J.Crew" |

---

## Data Quality Issues

### 1. **Scraped junk in description**
- **Row 10 (BM842)**: Full JavaScript/Boomerang performance script (~2KB) in `description`
- **Fix**: Drop rows where description contains `!function`, `BOOMR`, `go-mpulse`, etc., or truncate/sanitize

### 2. **Wrong care_instructions**
- **"Careers"**: Likely scraped from page footer/nav instead of care info
- **Long return policy**: Sometimes marketplace return instructions instead of care
- **Fix**: Treat "Careers" as empty; optionally flag long care text as possible junk

### 3. **Array formats (color_options, images_list)**
- Stored as JSON: `["Black", "Brown"]` and `["url1", "url2"]`
- **Target schema**: Pipe-separated `Black | Brown` and `url1 | url2`
- **Fix**: Parse JSON, join with ` | `

### 4. **Sale logic**
- `is_sale=True` when `price_original == price_current` (no real discount)
- **Fix**: Clear `price_original` when equal to current; set `is_sale=False`; recalc discount_percent

### 5. **Non-bag products**
- Rows 94–107: shorts, skirts, cardholders, kids' clothes (not bags)
- `category_breadcrumb` includes "girls > Bottoms", "men > Bags & Wallets", etc.
- **Fix**: Filter by `category_breadcrumb` containing "bags" if only bags desired, or add `leaf_category` and filter

### 6. **Missing fields**
- Many rows: empty description, dimensions, price_current, price_original
- **Fix**: Keep rows; use existing normalize/clean logic for empties

### 7. **leaf_category**
- Not present
- **Derive**: Last segment of `category_breadcrumb`, e.g. `"Home > women > bags"` → `"bags"`

### 8. **source**
- Not present
- **Add**: `"jcrew"` for all J.Crew rows

---

## Target Output Schema (products_normalized_combined)

Matches `COLLAPSED_COLUMNS` in `normalize_and_upload.py`:

| Column | J.Crew mapping |
|--------|----------------|
| product_id | product_id |
| source | "jcrew" |
| brand | brand |
| product_name | name |
| leaf_category | Last segment of category_breadcrumb |
| price | price_current |
| original_price | price_original (only if different from current) |
| discount_percent | Computed or empty |
| is_on_sale | is_sale (validated) |
| rating | average_rating |
| review_count | review_count |
| description | description (cleaned) |
| material_text | material_text |
| dimensions | dimensions |
| image_url | image_url |
| color_count | Len of parsed color_options |
| timestamp | scrape_date |
| product_url | product_url |

---

## Normalization Steps

1. **Detect J.Crew**: `jcrew.com` in product_url
2. **Parse arrays**: color_options, images_list from JSON → pipe-separated
3. **Clean description**: Remove JS/analytics blobs
4. **Clean care_instructions**: "Careers" → empty; optional sanity check on length
5. **Derive leaf_category**: From category_breadcrumb
6. **Fix sale fields**: price_original, is_sale, discount_percent
7. **Apply existing clean_row()**: dimensions, material, images filtering
8. **Collapse to user schema**: Same as Gap/Target
