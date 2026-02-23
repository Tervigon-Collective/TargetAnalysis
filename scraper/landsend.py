"""
Lands' End Bags - Product data extractor (search: q=bag).
Same behavior as J.Crew extractor: PLP + PDP, retries, rate limiting, currency to USD.
Output fields: availability_text, average_rating, brand, care_instructions, category_breadcrumb,
color_options, description, dimensions, image_count, image_url, images_list, is_sale, material_text,
name, price_currency, price_current, price_original, product_id, product_url, promo_excluded,
rating_count, review_count, scrape_date, sku, sold_shipped_by.
"""

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

from bs4 import BeautifulSoup

try:
    from playwright.sync_api import sync_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

BASE_URL = "https://www.landsend.com"
PLP_URL = "https://www.landsend.com/search?q=bag"
PLP_PAGE_DELAY_SEC = 1.0
PLP_MAX_PAGES = 20
DEFAULT_CURRENCY = "USD"
HEADLESS = True

PRICE_PATTERN_USD = re.compile(r"(?:USD\s*|\$)\s*([\d,.]+)")
PRICE_PATTERN_INR = re.compile(r"(?:INR|₹)\s*([\d,.]+)")
USD_TO_INR = 90.92
OUTPUT_CURRENCY = "USD"
PRICE_PATTERN = PRICE_PATTERN_USD

# PDP wait
PDP_WAIT_SELECTORS = ".product-detail, .product-name, [data-product-id], .breadcrumb, .product-price"
PDP_WAIT_TIMEOUT_MS = 18000
PDP_EXTRA_WAIT_SEC = 1.0

ACCESS_DENIED_MARKERS = ("Access Denied", "Reference #", "errors.edgesuite.net", "you don't have permission")
ACCESS_DENIED_WAIT_SEC = 10
MAX_ACCESS_DENIED_RETRIES = 10


class AccessDeniedError(Exception):
    pass


OUTPUT_FIELDS = [
    "availability_text", "average_rating", "brand", "care_instructions", "category_breadcrumb",
    "color_options", "description", "dimensions", "image_count", "image_url", "images_list",
    "is_sale", "material_text", "name", "price_currency", "price_current", "price_original",
    "product_id", "product_url", "promo_excluded", "rating_count", "review_count",
    "scrape_date", "sku", "sold_shipped_by",
]


def _text(el):
    if el is None:
        return ""
    return (el.get_text(strip=True) or "").strip()


def _parse_price_and_currency(text: str) -> tuple:
    if not text or not isinstance(text, str):
        return None, None
    text = text.strip()
    m = PRICE_PATTERN_INR.search(text)
    if m:
        try:
            val = float(m.group(1).replace(",", ""))
            return (int(val) if val == int(val) else val), "INR"
        except ValueError:
            pass
    m = PRICE_PATTERN_USD.search(text)
    if m:
        try:
            val = float(m.group(1).replace(",", ""))
            return (int(val) if val == int(val) else val), "USD"
        except ValueError:
            pass
    # Bare number (e.g. $59.50 without symbol in data)
    m = re.search(r"^\s*([\d,.]+)\s*$", text)
    if m:
        try:
            val = float(m.group(1).replace(",", ""))
            return (int(val) if val == int(val) else val), "USD"
        except ValueError:
            pass
    return None, None


def _convert_price_to(value: float, from_currency: str, to_currency: str):
    if from_currency == to_currency or value is None:
        return value
    if from_currency == "USD" and to_currency == "INR":
        return round(value * USD_TO_INR, 2)
    if from_currency == "INR" and to_currency == "USD":
        return round(value / USD_TO_INR, 2)
    return None


def _is_access_denied(html: str) -> bool:
    if not html or not isinstance(html, str):
        return False
    h = html.lower()
    return any(marker.lower() in h for marker in ACCESS_DENIED_MARKERS)


def _is_likely_color_name(text: str) -> bool:
    if not text or not isinstance(text, str) or len(text) > 80:
        return False
    t = text.strip().lower()
    if not t or PRICE_PATTERN_USD.search(text) or PRICE_PATTERN_INR.search(text):
        return False
    if "price" in t or "duties" in t or "includes" in t or "inr" in t or "usd" in t or "$" in text or "₹" in text:
        return False
    if re.search(r"^\d+[\d,.]*$", t):
        return False
    return True


def _attr(el, name, default=""):
    if el is None:
        return default
    return el.get(name, default) or default


def parse_rating_aria(aria_label):
    if not aria_label:
        return None
    m = re.search(r"([\d.]+)\s*out of\s*5", aria_label, re.I)
    return float(m.group(1)) if m else None


def parse_review_count(text):
    if not text:
        return None
    if isinstance(text, str):
        text = text.strip()
    m = re.search(r"(\d+)\s*reviews?", str(text), re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"^(\d+)$", str(text).strip())
    return int(m.group(1)) if m else None


def extract_plp_products(soup: BeautifulSoup) -> list:
    """Extract product entries from Lands' End PLP (search/category) page."""
    products = []
    # Common patterns: product grid, tiles, or list of links to /products/
    grid = (
        soup.find("ul", class_=re.compile(r"product-grid|product-list|search-result", re.I))
        or soup.find("div", class_=re.compile(r"product-grid|product-tiles|search-results", re.I))
        or soup.find(attrs={"data-testid": re.compile(r"product", re.I)})
    )
    if not grid:
        grid = soup

    # Find product links: Lands' End uses /products/... or show= in URL
    product_links = []
    for a in grid.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if "/products/" in href or "show=" in href or re.search(r"/id_\d+", href):
            full = urljoin(BASE_URL, href)
            if full not in {x.get("product_url") for x in products} and "landsend.com" in full:
                product_links.append((a, full))

    # Also try tile/card containers (SFCC often uses .product-tile, .tile)
    tiles = grid.find_all(
        "li",
        class_=re.compile(r"product-tile|product-cell|tile|search-result-item", re.I),
    ) or grid.find_all("div", class_=re.compile(r"product-tile|product-cell|tile", re.I))

    if tiles:
        for tile in tiles:
            item = {}
            link = tile.find("a", href=re.compile(r"/products/|show=|/id_\d+"))
            if not link and tile.name == "a" and tile.get("href"):
                link = tile
            href = _attr(link, "href") if link else ""
            item["product_url"] = urljoin(BASE_URL, href) if href else ""
            if not item["product_url"] or "landsend.com" not in item["product_url"]:
                continue
            item["product_id"] = _attr(tile, "data-product-id") or _attr(tile, "data-pid") or _attr(link, "data-product-id") or ""
            item["sku"] = _attr(tile, "data-sku") or _attr(tile, "data-product-id") or ""
            name_el = tile.find(class_=re.compile(r"product-name|product-title|name|title", re.I)) or link
            item["name"] = _text(name_el) or _attr(link, "title") or ""

            price_el = tile.find(class_=re.compile(r"price|product-price|sale-price|standard-price", re.I))
            if price_el:
                raw = _text(price_el)
                val, curr = _parse_price_and_currency(raw)
                if val is not None:
                    item["price_current"] = int(val) if isinstance(val, float) and val == int(val) else round(float(val), 2)
                    item["price_currency"] = curr or DEFAULT_CURRENCY
            was_el = tile.find(class_=re.compile(r"was-price|strikethrough|original-price|list-price", re.I))
            item["is_sale"] = bool(was_el)
            if was_el:
                val, curr = _parse_price_and_currency(_text(was_el))
                if val is not None:
                    item["price_original"] = int(val) if isinstance(val, float) and val == int(val) else round(float(val), 2)
            if item.get("price_original") is None:
                item["price_original"] = item.get("price_current")

            img = tile.find("img", src=True)
            if img:
                item["image_url"] = urljoin(BASE_URL, img.get("src") or "")

            color_opts = []
            for sw in tile.find_all(class_=re.compile(r"swatch|color", re.I)):
                for img in sw.find_all("img", alt=True):
                    alt = (_attr(img, "alt") or "").strip()
                    if alt and _is_likely_color_name(alt):
                        color_opts.append(alt)
            item["color_options"] = color_opts if color_opts else None

            rating_el = tile.find(attrs={"aria-label": re.compile(r"rating|star", re.I)})
            if rating_el:
                item["average_rating"] = parse_rating_aria(_attr(rating_el, "aria-label"))
            item["rating_count"] = parse_review_count(_attr(tile, "data-review-count")) or None
            item["review_count"] = item.get("rating_count")
            products.append(item)

    # If no tiles found, build from product links only
    if not products and product_links:
        seen_urls = set()
        for a, full_url in product_links:
            if full_url in seen_urls:
                continue
            seen_urls.add(full_url)
            item = {"product_url": full_url, "name": _text(a) or _attr(a, "title") or "", "product_id": "", "sku": ""}
            # SKU from URL: /products/.../id_123 or show=123
            m = re.search(r"/id_(\d+)|show=(\d+)", full_url)
            if m:
                item["product_id"] = m.group(1) or m.group(2) or ""
                item["sku"] = item["product_id"]
            price_el = a.find_parent(class_=re.compile(r"tile|product")) and a.find_parent().find(class_=re.compile(r"price", re.I))
            if price_el:
                val, curr = _parse_price_and_currency(_text(price_el))
                if val is not None:
                    item["price_current"] = round(float(val), 2)
                    item["price_currency"] = curr or DEFAULT_CURRENCY
            item["price_original"] = item.get("price_current")
            item["is_sale"] = False
            item["image_url"] = None
            item["color_options"] = None
            item["average_rating"] = None
            item["rating_count"] = None
            item["review_count"] = None
            products.append(item)

    return products


def extract_pdp_details(soup: BeautifulSoup, product_url: str) -> dict:
    """Extract full product details from a Lands' End PDP."""
    out = {}

    bread = soup.find("nav", attrs={"aria-label": re.compile(r"breadcrumb", re.I)}) or soup.find(class_=re.compile(r"breadcrumb", re.I))
    if bread:
        parts = [_text(a) for a in bread.find_all("a")]
        out["category_breadcrumb"] = " > ".join(parts) if parts else None
    else:
        out["category_breadcrumb"] = None

    name_el = soup.find("h1", class_=re.compile(r"product-name|product-title|name", re.I)) or soup.find("h1")
    if name_el:
        out["name"] = _text(name_el)

    # Price – convert all to USD
    price_el = soup.find(class_=re.compile(r"product-price|price|sale-price|standard-price", re.I))
    if price_el:
        full_price_text = price_el.get_text(separator=" ", strip=True)

        def _value_in_usd(val, curr):
            if val is None:
                return None, False
            v = float(val) if isinstance(val, (int, float)) else val
            c = curr or None
            if c == "USD":
                return v, True
            if c == "INR":
                converted = _convert_price_to(v, "INR", "USD")
                return converted, converted is not None
            if c is None:
                return (v, True) if v < 2000 else (_convert_price_to(v, "INR", "USD"), True)
            return None, False

        sale_el = price_el.find(class_=re.compile(r"sale|discount", re.I))
        was_el = price_el.find(class_=re.compile(r"was|strikethrough|list-price", re.I))
        reg_el = price_el.find(class_=re.compile(r"standard|regular|current", re.I)) or price_el
        if sale_el:
            raw = _text(sale_el)
            val, curr = _parse_price_and_currency(raw)
            if val is not None:
                v, ok = _value_in_usd(val, curr)
                if ok:
                    out["price_current"] = v
                    out["price_currency"] = OUTPUT_CURRENCY
                    out["is_sale"] = True
        if was_el and out.get("is_sale"):
            raw = _text(was_el)
            val, curr = _parse_price_and_currency(raw)
            if val is not None:
                v, ok = _value_in_usd(val, curr)
                if ok:
                    out["price_original"] = v
        if out.get("price_current") is None:
            raw = _text(reg_el)
            val, curr = _parse_price_and_currency(raw)
            v, ok = _value_in_usd(val, curr)
            if ok:
                out["price_current"] = v
                out["price_currency"] = OUTPUT_CURRENCY
        if out.get("price_current") is None:
            val, curr = _parse_price_and_currency(full_price_text)
            v, ok = _value_in_usd(val, curr)
            if ok:
                out["price_current"] = v
                out["price_currency"] = OUTPUT_CURRENCY
        cur_val, orig_val = out.get("price_current"), out.get("price_original")
        if cur_val is not None and orig_val is not None:
            if orig_val > 10000 and cur_val < 10000:
                out["price_original"] = None
            elif cur_val > 10000 and orig_val < 10000:
                out["price_current"] = None
    if out.get("price_currency") is None or not out["price_currency"]:
        out["price_currency"] = OUTPUT_CURRENCY

    desc_section = soup.find(class_=re.compile(r"product-description|description|product-details", re.I))
    if desc_section:
        out["description"] = _text(desc_section)
        tech = desc_section.find(class_=re.compile(r"material|features|details", re.I))
        if tech:
            out["material_text"] = _text(tech)
    else:
        out["description"] = None
        out["material_text"] = None

    dim_el = soup.find(string=re.compile(r"dimension|size|measure", re.I))
    if dim_el and hasattr(dim_el, "parent"):
        out["dimensions"] = _text(dim_el.parent)
    else:
        out["dimensions"] = None

    images_list = []
    gallery = soup.find(class_=re.compile(r"gallery|product-image|carousel", re.I)) or soup.find("main")
    if gallery:
        for img in gallery.find_all("img", src=True):
            src = img.get("src") or ""
            if "product" in src.lower() or "image" in src.lower() or "photo" in src.lower() or re.search(r"/\d+\.", src):
                full = urljoin(BASE_URL, src)
                if full not in images_list:
                    images_list.append(full)
    out["images_list"] = images_list if images_list else None
    out["image_count"] = len(images_list)
    out["image_url"] = images_list[0] if images_list else None

    color_opts = []
    color_sec = soup.find(class_=re.compile(r"color|swatch|variant", re.I))
    if color_sec:
        for img in color_sec.find_all("img", alt=True):
            alt = (_attr(img, "alt") or "").strip()
            if alt and _is_likely_color_name(alt):
                color_opts.append(alt)
        for btn in color_sec.find_all(["button", "span"], class_=re.compile(r"color|swatch", re.I)):
            label = _text(btn)
            if label and _is_likely_color_name(label):
                color_opts.append(label)
    out["color_options"] = list(dict.fromkeys(color_opts)) if color_opts else None

    add_btn = soup.find("button", string=re.compile(r"add to bag|add to cart|buy", re.I)) or soup.find(attrs={"data-add-to-cart": True})
    if add_btn:
        out["availability_text"] = "Out of stock" if add_btn.get("disabled") is not None else "In stock"
    else:
        out["availability_text"] = None

    out["brand"] = "Lands' End"

    care = soup.find(string=re.compile(r"care|instruction", re.I))
    out["care_instructions"] = _text(care.parent) if care and hasattr(care, "parent") else None

    rating_el = soup.find(attrs={"aria-label": re.compile(r"rating|star", re.I)})
    if rating_el:
        out["average_rating"] = parse_rating_aria(_attr(rating_el, "aria-label"))
    for s in soup.find_all(string=re.compile(r"\d+\s*reviews?", re.I)):
        out["review_count"] = parse_review_count(s)
        if out.get("review_count") is not None:
            break
    if out.get("rating_count") is None and out.get("review_count") is not None:
        out["rating_count"] = out["review_count"]

    out["promo_excluded"] = None
    out["sold_shipped_by"] = "Lands' End"
    out["product_url"] = product_url

    # SKU from URL
    m = re.search(r"/id_(\d+)|show=(\d+)", product_url)
    if m and not out.get("product_id"):
        out["product_id"] = m.group(1) or m.group(2) or ""
        out["sku"] = out["product_id"]

    return out


def merge_plp_and_pdp(plp_item: dict, pdp_item: dict) -> dict:
    merged = {f: None for f in OUTPUT_FIELDS}
    merged["brand"] = "Lands' End"
    merged["sold_shipped_by"] = "Lands' End"
    merged["scrape_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for k in OUTPUT_FIELDS:
        v = pdp_item.get(k) if pdp_item.get(k) is not None and pdp_item.get(k) != "" and pdp_item.get(k) != [] else plp_item.get(k)
        if v is not None and v != "" and v != []:
            merged[k] = v
    if merged.get("price_currency") == "INR":
        cur, orig = merged.get("price_current"), merged.get("price_original")
        if cur is not None:
            merged["price_current"] = _convert_price_to(float(cur), "INR", "USD")
        if orig is not None:
            merged["price_original"] = _convert_price_to(float(orig), "INR", "USD")
        merged["price_currency"] = OUTPUT_CURRENCY
    return {k: merged[k] for k in OUTPUT_FIELDS}


FETCH_RETRIES = 3
FETCH_RETRY_DELAY_SEC = 2
PDP_DELAY_SEC = 0.7
PDP_RETRY_DELAY_SEC = 3
PDP_FAILED_RETRIES = 2
PLP_SCROLL_STEPS = 15
PLP_SCROLL_PX = 700
PLP_SCROLL_PAUSE_SEC = 0.7
PLP_SCROLL_STABLE_CHECKS = 3
PLP_SCROLL_MAX_ITERATIONS = 25
PLP_FETCH_RETRIES = 2
PLP_PROXY_TRY_COUNT = 8

PROXY_LIST = []
DEFAULT_PROXY_LIST = [
    "http://136.49.32.180:8888",
    "http://72.56.59.17:61931",
    "http://104.238.30.58:63744",
    "http://72.56.59.62:63133",
    "http://72.56.59.23:61937",
    "http://104.238.30.86:63900",
    "http://104.238.30.37:59741",
    "http://72.56.59.56:63127",
    "http://104.238.30.91:63900",
    "http://104.238.30.50:59741",
    "http://104.238.30.40:59741",
    "http://104.238.30.45:59741",
    "http://104.238.30.38:59741",
    "http://104.238.30.39:59741",
    "http://20.210.113.32:8123",
    "http://72.56.50.17:59787",
]
US_PROXIES_API = "https://api.proxyscrape.com/v2/?request=displayproxies&protocol=http&country=US&timeout=5000"

DEBUG = True
# Set False to use direct connection (no proxies).
USE_PROXIES = False

BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
BROWSER_VIEWPORT = {"width": 1920, "height": 1080}
BROWSER_LOCALE = "en-US"


def _log(level: str, msg: str, *args) -> None:
    print(f"[{level}] {msg % args if args else msg}")


def get_us_proxies() -> list:
    global PROXY_LIST
    if PROXY_LIST:
        return PROXY_LIST
    single = os.environ.get("PROXY_URL", "").strip()
    if single:
        return [single if "://" in single else f"http://{single}"]
    many = os.environ.get("PROXIES", "").strip()
    if many:
        out = []
        for p in many.split(","):
            p = p.strip()
            if p:
                out.append(p if "://" in p else f"http://{p}")
        return out
    if DEFAULT_PROXY_LIST:
        return list(DEFAULT_PROXY_LIST)
    try:
        import urllib.request
        req = urllib.request.Request(US_PROXIES_API, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            lines = [ln.strip().decode("utf-8", errors="ignore") for ln in r if ln.strip()]
        return [f"http://{ln}" if "://" not in ln else ln for ln in lines if ln][:50]
    except Exception:
        return []


def fetch_html(url: str, use_playwright: bool = True, wait_for_selector: str | None = None, scroll_to_load_lazy: bool = False, retries: int = FETCH_RETRIES, proxy: str | None = None) -> str:
    last_error = None
    max_attempts = retries + MAX_ACCESS_DENIED_RETRIES
    for attempt in range(max_attempts):
        try:
            return _fetch_html_once(url, use_playwright, wait_for_selector, scroll_to_load_lazy, proxy)
        except AccessDeniedError as e:
            last_error = e
            if attempt < max_attempts - 1:
                time.sleep(ACCESS_DENIED_WAIT_SEC)
            else:
                raise
        except Exception as e:
            last_error = e
            if attempt < retries - 1:
                time.sleep(FETCH_RETRY_DELAY_SEC)
            else:
                raise last_error
    raise last_error


def fetch_plp_html(url: str, proxy: str | None = None, proxy_list: list | None = None) -> str:
    if not HAS_PLAYWRIGHT:
        return fetch_html(url, use_playwright=False, proxy=proxy)
    last_error = None
    last_html = ""
    last_count = 0
    proxies_to_try = [proxy] if proxy else (list(proxy_list[:PLP_PROXY_TRY_COUNT]) if proxy_list else [None])
    if not proxies_to_try:
        proxies_to_try = [None]
    for proxy_index, current_proxy in enumerate(proxies_to_try):
        launch_opts = {"headless": HEADLESS}
        if current_proxy:
            launch_opts["proxy"] = {"server": current_proxy}
        for plp_attempt in range(PLP_FETCH_RETRIES):
            for _ in range(MAX_ACCESS_DENIED_RETRIES):
                try:
                    with sync_playwright() as p:
                        browser = p.chromium.launch(**launch_opts)
                        ctx = browser.new_context(user_agent=BROWSER_USER_AGENT, viewport=BROWSER_VIEWPORT, locale=BROWSER_LOCALE)
                        page = ctx.new_page()
                        page.set_default_timeout(30000)
                        page.goto(url, wait_until="domcontentloaded", timeout=90000)
                        # Wait for product content (flexible)
                        try:
                            page.wait_for_selector("a[href*='/products/'], .product-tile, .product-grid, .search-result", state="attached", timeout=25000)
                        except Exception:
                            pass
                        stable, prev_count, iteration = 0, -1, 0
                        for iteration in range(PLP_SCROLL_MAX_ITERATIONS):
                            count = page.evaluate("""document.querySelectorAll('a[href*="/products/"], .product-tile, [class*="product-tile"]').length""")
                            if count == prev_count:
                                stable += 1
                                if stable >= PLP_SCROLL_STABLE_CHECKS:
                                    break
                            else:
                                stable = 0
                            prev_count = count
                            page.evaluate(f"window.scrollBy(0, {PLP_SCROLL_PX})")
                            time.sleep(PLP_SCROLL_PAUSE_SEC)
                        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                        time.sleep(2)
                        page.evaluate("window.scrollTo(0, 0)")
                        time.sleep(1)
                        last_html = page.content()
                        last_count = page.evaluate("""document.querySelectorAll('a[href*="/products/"], .product-tile, [class*="product-tile"]').length""")
                        ctx.close()
                        browser.close()
                        if _is_access_denied(last_html):
                            raise AccessDeniedError("Access Denied")
                        if last_count > 0:
                            return last_html
                        break
                except AccessDeniedError:
                    time.sleep(ACCESS_DENIED_WAIT_SEC)
                except Exception as e:
                    last_error = e
                    err_str = str(e).upper()
                    if "TUNNEL" in err_str or "ERR_PROXY" in err_str or "CONNECTION_FAILED" in err_str or "ERR_TIMED_OUT" in err_str:
                        break
                    break
        if last_html:
            return last_html
    raise last_error or RuntimeError("PLP fetch failed")


def _fetch_html_once(url: str, use_playwright: bool, wait_for_selector: str | None, scroll_to_load_lazy: bool, proxy: str | None = None) -> str:
    if use_playwright and HAS_PLAYWRIGHT:
        launch_opts = {"headless": HEADLESS}
        if proxy:
            launch_opts["proxy"] = {"server": proxy}
        with sync_playwright() as p:
            browser = p.chromium.launch(**launch_opts)
            ctx = browser.new_context(user_agent=BROWSER_USER_AGENT, viewport=BROWSER_VIEWPORT, locale=BROWSER_LOCALE)
            page = ctx.new_page()
            page.set_default_timeout(max(30000, PDP_WAIT_TIMEOUT_MS + 5000))
            page.goto(url, wait_until="load", timeout=90000)
            if wait_for_selector:
                try:
                    page.wait_for_selector(PDP_WAIT_SELECTORS, timeout=PDP_WAIT_TIMEOUT_MS)
                except Exception:
                    pass
                time.sleep(PDP_EXTRA_WAIT_SEC)
            if scroll_to_load_lazy:
                for _ in range(PLP_SCROLL_STEPS):
                    page.evaluate(f"window.scrollBy(0, {PLP_SCROLL_PX})")
                    time.sleep(PLP_SCROLL_PAUSE_SEC)
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                time.sleep(2)
            content = page.content()
            ctx.close()
            browser.close()
            if _is_access_denied(content):
                raise AccessDeniedError("Page returned Access Denied")
            return content
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": BROWSER_USER_AGENT})
    if proxy:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
        content = opener.open(req, timeout=30).read().decode()
    else:
        content = urllib.request.urlopen(req, timeout=30).read().decode()
    if _is_access_denied(content):
        raise AccessDeniedError("Page returned Access Denied")
    return content


def _build_plp_page_url(base_url: str, page: int, per_page: int = 96) -> str:
    parsed = urlparse(base_url)
    q = parse_qs(parsed.query, keep_blank_values=True)
    q["page"] = [str(page)]
    q["pageQuantity"] = [str(per_page)]
    return urlunparse(parsed._replace(query=urlencode(q, doseq=True)))


def run(
    plp_url: str = PLP_URL,
    output_json: str = "landsend_bags_products.json",
    output_csv: str = "landsend_bags_products.csv",
    use_playwright: bool = True,
    max_products: int | None = None,
    use_us_proxies: bool = True,
    reuse_browser: bool = True,
):
    global PROXY_LIST
    _log("INFO", "Starting Lands' End extractor (playwright=%s, headless=%s)", use_playwright, HEADLESS)
    use_browser = use_playwright and HAS_PLAYWRIGHT
    proxy_list = []
    if use_us_proxies:
        proxy_list = get_us_proxies()
        if not proxy_list:
            PROXY_LIST = []
            proxy_list = get_us_proxies()
        if not proxy_list:
            raise SystemExit("US proxies required but none available. Set PROXY_URL or PROXIES.")
        _log("INFO", "Using %d proxy(ies)", len(proxy_list))
    else:
        _log("INFO", "No proxies; using direct connection")
    plp_proxy = proxy_list[0] if proxy_list else None

    if use_browser and reuse_browser:
        with sync_playwright() as p:
            launch_opts = {"headless": HEADLESS}
            if plp_proxy:
                launch_opts["proxy"] = {"server": plp_proxy}
            browser = p.chromium.launch(**launch_opts)
            ctx = browser.new_context(user_agent=BROWSER_USER_AGENT, viewport=BROWSER_VIEWPORT, locale=BROWSER_LOCALE)
            try:
                all_products = []
                seen_urls = set()
                page_num = 1
                while page_num <= PLP_MAX_PAGES:
                    page_url = _build_plp_page_url(plp_url, page_num) if page_num > 1 else plp_url
                    _log("INFO", "Fetching PLP page %d: %s", page_num, page_url)
                    html = None
                    for access_denied_attempt in range(MAX_ACCESS_DENIED_RETRIES):
                        page = None
                        try:
                            page = ctx.new_page()
                            page.set_default_timeout(30000)
                            page.goto(page_url, wait_until="domcontentloaded", timeout=90000)
                            try:
                                page.wait_for_selector("a[href*='/products/'], .product-tile, .product-grid", state="attached", timeout=20000)
                            except Exception:
                                pass
                            for _ in range(PLP_SCROLL_MAX_ITERATIONS):
                                page.evaluate(f"window.scrollBy(0, {PLP_SCROLL_PX})")
                                time.sleep(PLP_SCROLL_PAUSE_SEC)
                            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                            time.sleep(1)
                            html = page.content()
                            page.close()
                            if _is_access_denied(html):
                                raise AccessDeniedError("Access Denied")
                            break
                        except AccessDeniedError:
                            if page is not None and not page.is_closed():
                                try:
                                    page.close()
                                except Exception:
                                    pass
                            if access_denied_attempt < MAX_ACCESS_DENIED_RETRIES - 1:
                                _log("INFO", "Access Denied; waiting %ss and retrying (%d/%d)...", ACCESS_DENIED_WAIT_SEC, access_denied_attempt + 1, MAX_ACCESS_DENIED_RETRIES)
                                time.sleep(ACCESS_DENIED_WAIT_SEC)
                            else:
                                _log("DEBUG", "PLP error: Access Denied (max retries reached)")
                                html = None
                                break
                        except Exception as e:
                            _log("DEBUG", "PLP error: %s", e)
                            if page is not None and not page.is_closed():
                                try:
                                    page.close()
                                except Exception:
                                    pass
                            break
                    if html is None:
                        break
                    soup = BeautifulSoup(html, "html.parser")
                    products = extract_plp_products(soup)
                    _log("INFO", "PLP page %d: %d items extracted", page_num, len(products))
                    if not products:
                        break
                    new_count = 0
                    for pr in products:
                        u = (pr.get("product_url") or "").strip()
                        if u and u not in seen_urls:
                            seen_urls.add(u)
                            all_products.append(pr)
                            new_count += 1
                    _log("INFO", "PLP page %d: %d new (total %d)", page_num, new_count, len(all_products))
                    if page_num > 1 and new_count == 0:
                        break
                    page_num += 1
                    if page_num <= PLP_MAX_PAGES:
                        time.sleep(PLP_PAGE_DELAY_SEC)
                plp_products = all_products
                if max_products:
                    plp_products = plp_products[:max_products]
                print(f"Found {len(plp_products)} products on PLP.")

                pdp_cache = {}
                results = []
                failed_pdp_urls = set()
                failed_plp_indices = []
                for i, plp_item in enumerate(plp_products):
                    url = (plp_item.get("product_url") or "").strip()
                    if not url:
                        results.append(merge_plp_and_pdp(plp_item, {}))
                        continue
                    if url in pdp_cache:
                        results.append(merge_plp_and_pdp(plp_item, pdp_cache[url]))
                        continue
                    print(f"  [{i+1}/{len(plp_products)}] PDP: {plp_item.get('name', url)[:50]}...")
                    if i > 0:
                        time.sleep(PDP_DELAY_SEC)
                    try:
                        page = ctx.new_page()
                        page.set_default_timeout(25000)
                        page.goto(url, wait_until="load", timeout=60000)
                        try:
                            page.wait_for_selector(PDP_WAIT_SELECTORS, timeout=PDP_WAIT_TIMEOUT_MS)
                        except Exception:
                            pass
                        time.sleep(PDP_EXTRA_WAIT_SEC)
                        pdp_html = page.content()
                        page.close()
                        if _is_access_denied(pdp_html):
                            raise AccessDeniedError("Access Denied")
                        pdp_soup = BeautifulSoup(pdp_html, "html.parser")
                        pdp_item = extract_pdp_details(pdp_soup, url)
                        pdp_cache[url] = pdp_item
                        results.append(merge_plp_and_pdp(plp_item, pdp_item))
                    except AccessDeniedError:
                        failed_pdp_urls.add(url)
                        failed_plp_indices.append((i, plp_item))
                        results.append(merge_plp_and_pdp(plp_item, {}))
                    except Exception as e:
                        _log("DEBUG", "PDP failed: %s", e)
                        failed_pdp_urls.add(url)
                        failed_plp_indices.append((i, plp_item))
                        results.append(merge_plp_and_pdp(plp_item, {}))

                for retry_round in range(PDP_FAILED_RETRIES):
                    if not failed_pdp_urls:
                        break
                    still_failed = set()
                    for i, plp_item in failed_plp_indices:
                        url = (plp_item.get("product_url") or "").strip()
                        if not url or url not in failed_pdp_urls:
                            continue
                        time.sleep(PDP_RETRY_DELAY_SEC)
                        try:
                            page = ctx.new_page()
                            page.goto(url, wait_until="load", timeout=60000)
                            time.sleep(PDP_EXTRA_WAIT_SEC)
                            pdp_html = page.content()
                            page.close()
                            if _is_access_denied(pdp_html):
                                raise AccessDeniedError("Access Denied")
                            pdp_soup = BeautifulSoup(pdp_html, "html.parser")
                            pdp_item = extract_pdp_details(pdp_soup, url)
                            pdp_cache[url] = pdp_item
                            failed_pdp_urls.discard(url)
                            results[i] = merge_plp_and_pdp(plp_item, pdp_item)
                        except Exception:
                            still_failed.add(url)
                    failed_pdp_urls = still_failed

                for r in results:
                    r["scrape_date"] = r.get("scrape_date") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                Path(output_json).parent.mkdir(parents=True, exist_ok=True)
                out_results = [{k: r.get(k) for k in OUTPUT_FIELDS} for r in results]
                with open(output_json, "w", encoding="utf-8") as f:
                    json.dump(out_results, f, indent=2, ensure_ascii=False)
                if results:
                    import csv
                    with open(output_csv, "w", newline="", encoding="utf-8") as f:
                        w = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
                        w.writeheader()
                        for r in out_results:
                            row = {k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()}
                            w.writerow(row)
                print(f"Saved {len(results)} products to {output_json} and {output_csv}.")
                return results
            finally:
                ctx.close()
                browser.close()

    # Non-reuse path: fetch PLP with fetch_plp_html then each PDP with fetch_html
    all_products = []
    seen_urls = set()
    page = 1
    while page <= PLP_MAX_PAGES:
        page_url = _build_plp_page_url(plp_url, page) if page > 1 else plp_url
        _log("INFO", "Fetching PLP page %d", page)
        html = fetch_plp_html(page_url, proxy=plp_proxy, proxy_list=proxy_list if proxy_list else None) if use_browser else fetch_html(page_url, use_playwright=False, proxy=plp_proxy)
        soup = BeautifulSoup(html, "html.parser")
        products = extract_plp_products(soup)
        if not products:
            break
        new_count = 0
        for p in products:
            u = (p.get("product_url") or "").strip()
            if u and u not in seen_urls:
                seen_urls.add(u)
                all_products.append(p)
                new_count += 1
        if page > 1 and new_count == 0:
            break
        page += 1
        time.sleep(PLP_PAGE_DELAY_SEC)
    plp_products = all_products[:max_products] if max_products else all_products
    print(f"Found {len(plp_products)} products on PLP.")

    results = []
    failed_pdp_urls = set()
    failed_plp_indices = []
    for i, plp_item in enumerate(plp_products):
        url = (plp_item.get("product_url") or "").strip()
        if not url:
            results.append(merge_plp_and_pdp(plp_item, {}))
            continue
        pdp_proxy = proxy_list[i % len(proxy_list)] if proxy_list else None
        print(f"  [{i+1}/{len(plp_products)}] PDP: {plp_item.get('name', url)[:50]}...")
        if i > 0:
            time.sleep(PDP_DELAY_SEC)
        try:
            pdp_html = fetch_html(url, use_playwright=use_browser, wait_for_selector=PDP_WAIT_SELECTORS, retries=FETCH_RETRIES, proxy=pdp_proxy)
            pdp_soup = BeautifulSoup(pdp_html, "html.parser")
            pdp_item = extract_pdp_details(pdp_soup, url)
            results.append(merge_plp_and_pdp(plp_item, pdp_item))
        except Exception as e:
            failed_pdp_urls.add(url)
            failed_plp_indices.append((i, plp_item))
            results.append(merge_plp_and_pdp(plp_item, {}))
    for retry_round in range(PDP_FAILED_RETRIES):
        if not failed_pdp_urls:
            break
        still_failed = set()
        for i, plp_item in failed_plp_indices:
            url = (plp_item.get("product_url") or "").strip()
            if not url or url not in failed_pdp_urls:
                continue
            time.sleep(PDP_RETRY_DELAY_SEC)
            try:
                pdp_html = fetch_html(url, use_playwright=use_browser, wait_for_selector=PDP_WAIT_SELECTORS, proxy=proxy_list[i % len(proxy_list)] if proxy_list else None)
                pdp_soup = BeautifulSoup(pdp_html, "html.parser")
                pdp_item = extract_pdp_details(pdp_soup, url)
                failed_pdp_urls.discard(url)
                results[i] = merge_plp_and_pdp(plp_item, pdp_item)
            except Exception:
                still_failed.add(url)
        failed_pdp_urls = still_failed

    for r in results:
        r["scrape_date"] = r.get("scrape_date") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    out_results = [{k: r.get(k) for k in OUTPUT_FIELDS} for r in results]
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(out_results, f, indent=2, ensure_ascii=False)
    if results:
        import csv
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
            w.writeheader()
            for r in out_results:
                row = {k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()}
                w.writerow(row)
    print(f"Saved {len(results)} products to {output_json} and {output_csv}.")
    return results


if __name__ == "__main__":
    run(use_playwright=HAS_PLAYWRIGHT, max_products=None if DEBUG else None, use_us_proxies=USE_PROXIES)