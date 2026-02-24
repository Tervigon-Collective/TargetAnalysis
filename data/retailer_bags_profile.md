# Retailer Bags – Multi-Source Data Profile

Profiles for Gap, Abercrombie, and Aritzia bag CSVs, normalized to the canonical schema and uploaded via `normalize_and_upload.py`.

---

## Gap (gap_bag 1.csv)

| Aspect | Notes |
|--------|------|
| **Schema** | Canonical-style: availability_text, category_breadcrumb, color_options, description, dimensions, image_url, images_list, is_sale, material_text, name, price_currency, price_current, price_original, product_id, product_url, etc. |
| **product_id** | Numeric (e.g. 1169519002) |
| **product_url** | gap.com/browse/product.do?pid=... |
| **dimensions** | Often contains UI junk: "One Size Sold & shipped by Kibou Return by mail only. 1 Add to Bag" |
| **images_list** | Pipe-separated; some URLs are cookielaw/Logo junk |
| **material_text** | Sometimes STATE donation blurb (". For every bag sold, STATE donates...") |
| **care_instructions** | Sometimes "Customers also viewed" / "You may also like" junk |
| **leaf_category** | Empty (Gap doesn't populate); not derived |
| **Cleaning** | DIMENSIONS_UI_JUNK, CARE_JUNK_PATTERN, MATERIAL_JUNK_PATTERNS, NON_PRODUCT_IMAGE_PATTERNS |

---

## Abercrombie (abercrombie_bag 1.csv)

| Aspect | Notes |
|--------|------|
| **Schema** | Same canonical-style as Gap |
| **product_id** | Often empty in CSV; extracted from URL path `/p/slug-{id}` (e.g. bag-charm-61210335 → 61210335) |
| **product_url** | abercrombie.com/shop/wd/p/... |
| **color_options** | Pipe-separated ("blue \| red") |
| **images_list** | Pipe-separated |
| **price_current** | Values like 3265.81, 2332.72 – verify currency (may be cents/INR) |
| **leaf_category** | Derived from category_breadcrumb (e.g. "Bags & Bag Charms") |
| **Variants** | Same product_id, different color/variant URLs → dedup keeps first |
| **Cleaning** | Same as Gap; dimensions/dimensions_section usually clean |

---

## Aritzia (aritzia_bag 1.csv)

| Aspect | Notes |
|--------|------|
| **Schema** | Same canonical-style as Gap |
| **product_id** | Numeric (e.g. 123587, 130743) |
| **product_url** | aritzia.com/intl/en/product/... |
| **price_currency** | INR in sample (intl site) |
| **color_options** | Pipe-separated; sometimes "Color1 / Color2 \| Color3" format |
| **leaf_category** | Derived from category_breadcrumb (e.g. "Valet garment bag", "Awayday tote") |
| **brand** | Aritzia, Tna, The Super Puff™ |
| **dimensions** | Clean format ("59h x 140d centimetres...") |
| **Cleaning** | Same as Gap; generally clean |

---

## Target Output (products_normalized_combined)

| Column | Mapping |
|--------|---------|
| product_id | product_id (or extracted from URL for Abercrombie) |
| source | gap, abercrombie, aritzia |
| brand | brand |
| product_name | name |
| leaf_category | From breadcrumb (Aritzia, Abercrombie); empty (Gap) |
| price | price_current |
| original_price | price_original |
| discount_percent | Computed or empty |
| is_on_sale | is_sale |
| rating | average_rating |
| review_count | review_count |
| description | description |
| material_text | material_text |
| dimensions | dimensions (cleaned) |
| image_url | image_url |
| color_count | Count of color_options |
| timestamp | scrape_date |
| product_url | product_url |

---

## Commands

```powershell
# All three
python scripts/normalize_and_upload.py "data/gap_bag 1.csv" "data/abercrombie_bag 1.csv" "data/aritzia_bag 1.csv" --output-name products_multi_retailer

# Single source
python scripts/normalize_and_upload.py "data/gap_bag 1.csv"
python scripts/normalize_and_upload.py "data/aritzia_bag 1.csv"
python scripts/normalize_and_upload.py "data/abercrombie_bag 1.csv"
```
