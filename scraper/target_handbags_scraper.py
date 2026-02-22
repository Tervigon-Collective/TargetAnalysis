#!/usr/bin/env python3
"""
Simplified & cleaned Target Handbags Scraper

Extracts product data from Target's handbags category with pagination support.
Optionally visits detail pages for richer metadata.

Dedup: Keeps scraper_metadata.json in output-dir with already-fetched tcins.
Subsequent runs skip those products and only crawl new data. Use --fresh to disable.

Usage:
    python target_handbags_scraper.py --max-products 60 --details --output-dir ./data
    python target_handbags_scraper.py --concurrent-pages 40  # Parallel listing pages
    python target_handbags_scraper.py --concurrent-pdp 10   # Fetch 10 PDPs at once when --details
    python target_handbags_scraper.py --proxy-file output/proxies.csv.20260222_120247.bak  # Rotate on failure
    python target_handbags_scraper.py --fresh  # Force full re-crawl (ignore metadata dedup)
"""

import asyncio
import json
import logging
import random
import re
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from playwright.async_api import async_playwright, Page, BrowserContext
from bs4 import BeautifulSoup

# Metadata file for dedup – tracks already-fetched tcins so we skip re-crawling
METADATA_FILENAME = "scraper_metadata.json"

# Errors that indicate proxy/connection failure – rotate and retry
_PROXY_ROTATE_KEYWORDS = (
    "proxy", "ERR_PROXY", "ERR_CONNECTION", "ETIMEDOUT", "ENOTFOUND",
    "ECONNREFUSED", "net::ERR", "NS_BINDING", "Timeout", "timeout",
)

# ────────────────────────────────────────────────
# Logging
# ────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("target-scraper")


# ────────────────────────────────────────────────
# Data model
# ────────────────────────────────────────────────

@dataclass
class Product:
    """Product data structure – core identifiers + enriched PDP fields."""
    # ── Identifiers ──────────────────────────────────────────────────────────
    tcin: str
    title: str
    url: str

    # ── Brand / category ─────────────────────────────────────────────────────
    brand: str = ""
    breadcrumb: str = ""          # e.g. "Target > Accessories > Handbags"
    leaf_category: str = ""       # last crumb, e.g. "Shoulder Bags"

    # ── Pricing ──────────────────────────────────────────────────────────────
    price: float = 0.0
    original_price: float = 0.0
    discount_percent: float = 0.0
    is_on_sale: bool = False

    # ── Ratings ──────────────────────────────────────────────────────────────
    rating: float = 0.0
    review_count: int = 0

    # ── Engagement signals ───────────────────────────────────────────────────
    bought_recently: str = ""     # e.g. "50+ bought in past week"
    is_new: bool = False

    # ── Variants / colors ────────────────────────────────────────────────────
    colors: List[str] = field(default_factory=list)
    selected_color: str = ""

    # ── Media ────────────────────────────────────────────────────────────────
    images: List[str] = field(default_factory=list)

    # ── Content ──────────────────────────────────────────────────────────────
    description: str = ""
    feature_bullets: List[str] = field(default_factory=list)  # highlight bullets

    # ── Derived spec fields (quick-access) ───────────────────────────────────
    material: str = ""            # Shell Material
    dimensions: Dict[str, str] = field(default_factory=dict)  # parsed dim map
    dimensions_raw: str = ""      # e.g. "5.59 Inches (H) x 9.76 Inches (W) x 2.16 Inches (D)"
    bag_structure: str = ""       # Structured / Unstructured
    interior_features: str = ""
    exterior_features: str = ""
    closure_type: str = ""        # Flap Closure, Zipper, etc.
    handle_type: str = ""         # Shoulder Strap, Single handle, etc.
    fabric_name: str = ""         # Jacquard Weave, Canvas, etc.
    care_instructions: str = ""   # Spot or Wipe Clean, etc.
    origin: str = ""              # Imported / Domestic

    # ── Administrative ───────────────────────────────────────────────────────
    upc: str = ""
    dpci: str = ""                # Target item number (e.g. 024-06-6038)
    specs: Dict[str, str] = field(default_factory=dict)  # full raw spec map
    in_stock: bool = True
    scraped_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['colors'] = ' | '.join(self.colors)
        d['images'] = ' | '.join(self.images[:10])
        d['feature_bullets'] = ' | '.join(self.feature_bullets)
        d['dimensions'] = json.dumps(self.dimensions, ensure_ascii=False)
        d['specs'] = json.dumps(self.specs, ensure_ascii=False)
        return d


# ────────────────────────────────────────────────
# Scraper core
# ────────────────────────────────────────────────

class TargetHandbagsScraper:
    # Target uses Nao (offset) for pagination; typically ~24 items per page
    ITEMS_PER_PAGE = 24

    def __init__(
        self,
        max_products: Optional[int] = None,
        delay_range: tuple[float, float] = (1.2, 3.1),
        headless: bool = True,
        devtools: bool = False,
        slow_mo: int = 0,
        verbose: bool = False,
        get_details: bool = False,
        output_dir: str = "./data",
        skip_dedup: bool = False,
        proxy: Optional[str] = None,
        proxy_file: Optional[str] = None,
        concurrent_listing_pages: int = 1,
        concurrent_pdp: int = 1,
    ):
        self.max_products = max_products
        self.delay_min, self.delay_max = delay_range
        self.concurrent_listing_pages = max(1, concurrent_listing_pages)
        self.concurrent_pdp = max(1, concurrent_pdp)
        self.headless = headless
        self.devtools = devtools
        self.slow_mo = slow_mo
        self.verbose = verbose
        self.get_details = get_details
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.skip_dedup = skip_dedup
        self.proxy = proxy
        self.proxy_pool: List[str] = []
        if proxy_file:
            self.proxy_pool = self._load_proxy_pool(proxy_file)
        elif proxy:
            self.proxy_pool = [proxy]
        self._proxy_index = 0

        if verbose:
            logger.setLevel(logging.DEBUG)

        self.products: List[Product] = []
        self.base_url = "https://www.target.com/c/handbags-purses-accessories/-/N-5xtbo"
        self._pw = None
        self._browser = None
        self._context = None
        self._fetched_tcins: set = set()  # Loaded from metadata for dedup

    async def _delay(self):
        await asyncio.sleep(random.uniform(self.delay_min, self.delay_max))

    def _load_metadata(self) -> None:
        """Load fetched tcins from metadata file for dedup."""
        path = self.output_dir / METADATA_FILENAME
        if not path.exists():
            logger.info("No metadata file found – starting fresh")
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._fetched_tcins = set(data.get("fetched_tcins", []))
            logger.info(f"Loaded {len(self._fetched_tcins)} already-fetched tcins from metadata (dedup enabled)")
        except Exception as e:
            logger.warning(f"Could not load metadata: {e} – starting fresh")

    def _save_metadata(self, new_tcins: List[str]) -> None:
        """Update metadata with newly fetched tcins."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / METADATA_FILENAME
        self._fetched_tcins.update(new_tcins)
        data = {
            "fetched_tcins": sorted(self._fetched_tcins),
            "last_updated": datetime.now().isoformat(),
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info(f"Updated metadata: {len(self._fetched_tcins)} total tcins tracked")

    def _load_proxy_pool(self, path: str) -> List[str]:
        """Load proxies from file: one per line, ip:port or http://ip:port."""
        p = Path(path).expanduser().resolve()
        if not p.exists():
            logger.warning(f"Proxy file not found: {p}")
            return []
        proxies = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "://" not in line:
                line = f"http://{line}"
            proxies.append(line)
        logger.info(f"Loaded {len(proxies)} proxies from {p}")
        return proxies

    def _get_current_proxy(self) -> Optional[str]:
        if not self.proxy_pool:
            return None
        return self.proxy_pool[self._proxy_index % len(self.proxy_pool)]

    def _rotate_proxy(self) -> bool:
        """Move to next proxy. Returns True if a different proxy is available."""
        if len(self.proxy_pool) <= 1:
            return False
        self._proxy_index += 1
        return self._proxy_index < len(self.proxy_pool)

    @staticmethod
    def _is_proxy_error(exc: BaseException) -> bool:
        msg = str(exc).lower()
        return any(kw in msg for kw in (s.lower() for s in _PROXY_ROTATE_KEYWORDS))

    def _build_listing_urls(self, num_pages: int) -> List[str]:
        """Build listing page URLs using Target's Nao (offset) pagination."""
        urls = []
        for i in range(num_pages):
            offset = i * self.ITEMS_PER_PAGE
            sep = "&" if "?" in self.base_url else "?"
            urls.append(f"{self.base_url}{sep}Nao={offset}")
        return urls

    async def _bring_to_front(self, page: Page, label: str):
        try:
            await page.bring_to_front()
        except Exception:
            logger.debug(f"Could not bring page to front ({label})")

    async def _init_browser(self) -> tuple[BrowserContext, Page]:
        self._pw = await async_playwright().start()

        # Debug mode: DevTools requires a headed browser.
        launch_headless = False if self.devtools else self.headless

        launch_kwargs: dict = {
            "headless": launch_headless,
            "devtools": self.devtools,
            "slow_mo": self.slow_mo or 0,
        }
        proxy = self._get_current_proxy()
        if proxy:
            # Playwright requires credentials as separate fields, NOT embedded
            # in the URL (http://user:pass@host:port causes ERR_INVALID_AUTH_CREDENTIALS).
            from urllib.parse import urlparse
            raw = proxy if "://" in proxy else f"http://{proxy}"
            parsed = urlparse(raw)
            server = f"{parsed.scheme}://{parsed.hostname}:{parsed.port}"
            proxy_dict: dict = {"server": server}
            if parsed.username:
                proxy_dict["username"] = parsed.username
            if parsed.password:
                proxy_dict["password"] = parsed.password
            launch_kwargs["proxy"] = proxy_dict
            logger.info(f"Using proxy: {server}" + (f" (authenticated as {parsed.username})" if parsed.username else ""))
        self._browser = await self._pw.chromium.launch(**launch_kwargs)

        context = await self._browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/128.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900},
            locale="en-US",
            ignore_https_errors=True,
            java_script_enabled=True,
            extra_http_headers={
                # Only set headers that are safe to send on EVERY request
                # (including cross-origin sub-resources).
                # Sec-Fetch-*, Upgrade-Insecure-Requests are navigation-only
                # headers; sending them on XHR/CSS/JS triggers CORS preflight
                # failures on Target's CDN (assets.targetimg1.com) which
                # blocks all Next.js bundles and prevents React from hydrating.
                "Accept-Language": "en-US,en;q=0.9",
                "Sec-Ch-Ua": '"Chromium";v="128", "Not;A=Brand";v="24", "Google Chrome";v="128"',
                "Sec-Ch-Ua-Mobile": "?0",
                "Sec-Ch-Ua-Platform": '"Windows"',
            },
        )
        page = await context.new_page()
        self._context = context

        if self.verbose or self.devtools:
            page.on("console", lambda msg: logger.info(f"PAGE CONSOLE: {msg.type}: {msg.text}"))
            page.on("pageerror", lambda err: logger.error(f"PAGE ERROR: {err}"))
        return context, page

    async def _close_browser(self):
        """Close browser and playwright; safe to call if already closed."""
        try:
            if getattr(self, "_context", None):
                await self._context.close()
                self._context = None
        except Exception:
            pass
        try:
            if self._browser is not None:
                await self._browser.close()
                self._browser = None
        except Exception:
            pass
        try:
            if self._pw is not None:
                await self._pw.stop()
                self._pw = None
        except Exception:
            pass

    # ─── Listing page helpers ────────────────────────────────────────────────

    async def _dismiss_modals(self, page: Page):
        """Close sign-in popups, location prompts, or cookie banners if present."""
        dismissals = [
            # Sign-in tooltip close button
            'button[aria-label="close"]',
            'button[data-test="closeButton"]',
            # Generic close / dismiss
            '[aria-label="Close"]',
            '[data-test="modal-close"]',
        ]
        for sel in dismissals:
            try:
                btn = page.locator(sel).first
                if await btn.is_visible(timeout=800):
                    await btn.click()
                    logger.debug(f"Dismissed modal via: {sel}")
                    await asyncio.sleep(0.3)
            except Exception:
                pass

    async def _screenshot_debug(self, page: Page, label: str):
        """Save a screenshot to output_dir when something goes wrong."""
        try:
            dest = self.output_dir / f"debug_{label}_{datetime.now().strftime('%H%M%S')}.png"
            await page.screenshot(path=str(dest), full_page=False)
            logger.info(f"Debug screenshot saved → {dest}")
        except Exception as e:
            logger.debug(f"Could not save debug screenshot: {e}")

    async def _wait_for_products(self, page: Page, timeout: int = 28000) -> bool:
        """Wait for product cards, retrying once if we hit a bot-challenge render."""
        selector = '[data-test="@web/ProductCard/title"]'
        try:
            await page.wait_for_selector(selector, timeout=timeout)
            return True
        except Exception:
            pass

        # Check for bot challenge indicators
        challenge = await page.evaluate("""
        () => {
            const text = document.body?.innerText || '';
            return text.includes('Access Denied') ||
                   text.includes('verify you are human') ||
                   document.querySelector('#px-captcha') !== null;
        }
        """)
        if challenge:
            logger.warning("Bot challenge detected – waiting 8 s before retry")
            await asyncio.sleep(8)
        else:
            # React might still be hydrating; give it a few more seconds
            logger.debug("Products not yet in DOM, waiting 5 s for late render…")
            await asyncio.sleep(5)

        # One retry
        try:
            await page.wait_for_selector(selector, timeout=15000)
            return True
        except Exception:
            return False

    async def _scrape_listing_page(self, page: Page, url: str) -> List[Product]:
        logger.info(f"Scraping → {url}")
        await self._bring_to_front(page, "listing")
        # "load" waits for the full load event (all scripts & stylesheets
        # referenced in HTML), giving React time to fully hydrate before we
        # check for product cards.
        await page.goto(url, wait_until="load", timeout=60000)
        
        # Extra wait for React hydration and client-side rendering
        await asyncio.sleep(3)

        # Dismiss any sign-in / location / cookie modals that block the grid
        await self._dismiss_modals(page)

        found = await self._wait_for_products(page)
        if not found:
            logger.warning("No product cards found on this page")
            await self._screenshot_debug(page, "no_products")
            return []

        # Additional wait after products are found to let images/prices render
        await asyncio.sleep(2)

        # Scroll all the way to the bottom so every lazy-loaded card appears
        await self._scroll_to_bottom(
            page,
            card_selector='[data-test="@web/ProductCard/title"]',
        )
        
        # Final wait after scroll completes to ensure all lazy content is rendered
        await asyncio.sleep(2)

        items = await page.evaluate("""
        () => {
            const cards = document.querySelectorAll('[data-test="@web/site-top-of-funnel/ProductCardWrapper"]');
            return Array.from(cards).map(card => {
                const link = card.querySelector('a[data-test="@web/ProductCard/title"]');
                if (!link) return null;

                const href = link.getAttribute('href') || '';
                const title = link.textContent?.trim() || '';

                // Current price
                const priceEl = card.querySelector('[data-test="current-price"]');
                const price = priceEl?.textContent?.trim() || '';

                // Original / regular price (shown when on sale)
                const origEl = card.querySelector('[data-test="regular-price"]') ||
                               card.querySelector('s') ||
                               card.querySelector('[class*="regular"]');
                const originalPrice = origEl?.textContent?.trim() || '';

                // Sale / badge signals
                const isSale = !!card.querySelector(
                    '[data-test="sale-badge"], [class*="saleBadge"], [data-test*="strikethrough"]'
                );
                const isNew = !!card.querySelector('[data-test*="new-badge"], [class*="newBadge"]');

                // Bought-recently badge
                const boughtEl = card.querySelector('[class*="boughtRecently"], [data-test*="boughtRecently"]');
                const boughtRecently = boughtEl?.textContent?.trim() || '';

                // Rating & review count if visible on card
                const ratingEl = card.querySelector('[data-test="ratings"], [class*="ratingCount"]');
                const ratingText = ratingEl?.textContent?.trim() || '';

                // Brand on card
                const brandEl = card.querySelector('[data-test*="brand"], [class*="BrandName"], [class*="brandName"]');
                const brand = brandEl?.textContent?.trim() || '';

                return { href, title, price, originalPrice, isSale, isNew, boughtRecently, ratingText, brand };
            }).filter(Boolean);
        }
        """)

        products = []
        for item in items:
            if not item['href']:
                continue
            full_url = "https://www.target.com" + item['href'] if item['href'].startswith("/") else item['href']
            tcin = re.search(r'/A-(\d+)', full_url)
            tcin = tcin.group(1) if tcin else ""

            prod = Product(
                tcin=tcin,
                title=item['title'],
                url=full_url,
                price=self._parse_price(item['price']),
                original_price=self._parse_price(item.get('originalPrice', '')),
                is_on_sale=bool(item.get('isSale')),
                is_new=bool(item.get('isNew')),
                bought_recently=item.get('boughtRecently', ''),
                brand=item.get('brand', ''),
            )
            # Parse rating/review from card text "X out of 5 stars with Y reviews"
            card_rating = item.get('ratingText', '')
            if card_rating:
                rm = re.search(r'([\d.]+)\s+out of\s+5', card_rating, re.I)
                if rm:
                    try:
                        prod.rating = float(rm.group(1))
                    except ValueError:
                        pass
                rv = re.search(r'with\s+([\d,]+)\s+(?:rating|review)', card_rating, re.I)
                if rv:
                    try:
                        prod.review_count = int(rv.group(1).replace(',', ''))
                    except ValueError:
                        pass
            products.append(prod)

            if self.max_products and len(self.products) + len(products) >= self.max_products:
                break

        logger.info(f"   → found {len(products)} products")
        return products

    async def _scroll_to_bottom(
        self,
        page: Page,
        card_selector: str = "",
        max_attempts: int = 30,
        network_wait_ms: int = 2000,
    ):
        """
        Scroll to the bottom, waiting just long enough for lazy-loaded DOM to
        appear.

        network_wait_ms:

          > 0  – used on listing pages; waits briefly for the XHR that an
                 infinite-scroll trigger fires after each scroll step.
          0    – used on static detail pages; skips the network wait entirely
                 (the page is already fully loaded; we only need images/specs
                 that are below-the-fold to enter the viewport and render).
        """
        prev_height = -1
        stable = 0

        for attempt in range(max_attempts):
            cur_height = await page.evaluate("""
                () => {
                    window.scrollTo({ top: document.body.scrollHeight, behavior: 'instant' });
                    return document.body.scrollHeight;
                }
            """)

            if network_wait_ms > 0:
                # Short pause so the scroll-triggered XHR can fire, then wait
                # for it to settle (best-effort; don't block long).
                await asyncio.sleep(0.4)
                try:
                    await page.wait_for_load_state("networkidle", timeout=network_wait_ms)
                except Exception:
                    pass
            else:
                # Detail page: already loaded – just let the browser paint the
                # newly-visible section before we re-check height.
                await asyncio.sleep(0.25)

            new_height = await page.evaluate("() => document.body.scrollHeight")

            if new_height == prev_height:
                stable += 1
                if stable >= 3:
                    logger.debug(f"Scroll stable after {attempt + 1} steps (height={new_height})")
                    break
            else:
                stable = 0
                prev_height = new_height
                if card_selector:
                    count = await page.evaluate(
                        f"""() => document.querySelectorAll('{card_selector}').length"""
                    )
                    logger.debug(f"  scroll {attempt + 1}: height={new_height}, cards={count}")
                    if self.max_products and count >= self.max_products:
                        break
                else:
                    logger.debug(f"  scroll {attempt + 1}: height={new_height}")

        # Return to top so pagination buttons / next-page logic can find them
        await page.evaluate("window.scrollTo(0, 0)")
        await asyncio.sleep(0.2)

    # ─── Detail page helpers ─────────────────────────────────────────────────

    def _parse_specs_text(self, spec_el) -> Dict[str, str]:
        """
        Parse spec section text into key/value pairs.

        Target uses two layouts:
          1. Key ends with colon, value is on next line:
               "Dimensions (Overall):\n5.59 Inches..."
          2. Key has no colon, value starts with ": " on next line (TCIN, UPC):
               "TCIN\n: 94780342"
        """
        if not spec_el:
            return {}
        raw = spec_el.get_text(separator="\n", strip=True)
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        parsed: Dict[str, str] = {}
        i = 0
        while i < len(lines):
            line = lines[i]
            # Layout 1: "Key:"
            if line.endswith(":"):
                key = line[:-1].strip()
                if key and i + 1 < len(lines):
                    next_line = lines[i + 1]
                    # make sure next line isn't itself a key
                    if not next_line.endswith(":") and not next_line.startswith(": "):
                        parsed[key] = next_line
                        i += 2
                        continue
            # Layout 2: value on next line starts with ": "
            elif i + 1 < len(lines) and lines[i + 1].startswith(": "):
                key = line.strip()
                val = lines[i + 1][2:].strip()   # strip leading ": "
                if key and val:
                    parsed[key] = val
                i += 2
                continue
            # Layout 3: "Key: Value" on a single line
            elif ":" in line:
                parts = line.split(":", 1)
                key, val = parts[0].strip(), parts[1].strip()
                if key and val:
                    parsed[key] = val
            i += 1
        return parsed

    def _apply_specs(self, product: Product, specs: Dict[str, str]):
        """
        Populate structured fields from the raw spec dict and store the full map.
        """
        product.specs.update(specs)
        for key, value in specs.items():
            kl = key.lower()
            if not value:
                continue
            # Material — must say "material" in the key ("shell" alone is too
            # broad; "Shell Color", "Outer Shell" etc. are different specs)
            if "material" in kl:
                product.material = value
            # Dimensions – store raw string AND per-axis dict
            if "dimension" in kl:
                product.dimensions_raw = value
                # parse "5.59 Inches (H) x 9.76 Inches (W) x 2.16 Inches (D)"
                for axis_match in re.finditer(
                    r'([\d.]+)\s+\w+\s*\(([^)]+)\)', value
                ):
                    product.dimensions[axis_match.group(2).upper()] = axis_match.group(1)
            elif any(d in kl for d in ["height", "width", "depth", "length"]):
                product.dimensions[key] = value
            # Bag structure — require "bag structure" exactly so that
            # "Interior Structure", "Frame Structure" etc. are not captured
            if "bag structure" in kl:
                product.bag_structure = value
            if "interior feature" in kl:
                product.interior_features = value
            if "exterior feature" in kl:
                product.exterior_features = value
            if "closure" in kl:
                product.closure_type = value
            if "handle" in kl:
                product.handle_type = value
            if "fabric" in kl:
                product.fabric_name = value
            if "care" in kl or "cleaning" in kl:
                product.care_instructions = value
            if "origin" in kl:
                product.origin = value
            if kl == "upc":
                product.upc = value
            if "dpci" in kl or "item number" in kl:
                product.dpci = value
            if kl == "tcin" and not product.tcin:
                product.tcin = value

    async def _enrich_with_details(self, page: Page, product: Product):
        logger.info(f"Enriching product detail: {product.tcin} - {product.title}")
        try:
            await self._bring_to_front(page, "detail")
            # "load" ensures all scripts are executed before we look for
            # product title, specs, and images.
            await page.goto(product.url, wait_until="load", timeout=60000)
            
            # Wait for React hydration
            await asyncio.sleep(3)
            
            await page.wait_for_selector('h1[data-test="product-title"]', timeout=20000)
            
            # Additional wait for dynamic content (images, specs, etc.)
            await asyncio.sleep(2)

            # Expand accordion sections if collapsed, scrolling them into
            # view first so Playwright can interact with them reliably.
            for label in ["Specifications", "About this item", "Highlights", "Details"]:
                try:
                    btn = page.locator(f'button:has-text("{label}")').first
                    if await btn.count() > 0:
                        await btn.scroll_into_view_if_needed(timeout=3000)
                        await asyncio.sleep(0.2)
                        expanded = await btn.get_attribute("aria-expanded")
                        if expanded != "true":
                            await btn.click()
                            await asyncio.sleep(0.5)
                except Exception:
                    pass

            # Scroll each product-data section into view in sequence so
            # lazy-loaded content renders without going past the product area.
            detail_sections = [
                '[data-test="product-title"]',
                '[data-test="product-price"]',
                '[data-test="item-details-description"]',
                '[data-test="item-details-specifications"]',
                '[data-test="item-details-highlights"]',
            ]
            for sel in detail_sections:
                try:
                    el = page.locator(sel).first
                    if await el.count() > 0:
                        await el.scroll_into_view_if_needed(timeout=3000)
                        await asyncio.sleep(0.15)
                except Exception:
                    pass

            html = await page.content()
            soup = BeautifulSoup(html, "html.parser")

            # ── Title ───────────────────────────────────────────────────────
            title_el = soup.select_one('h1[data-test="product-title"]')
            if title_el:
                product.title = title_el.get_text(strip=True)

            # ── Brand ───────────────────────────────────────────────────────
            brand_el = soup.select_one('a[data-test="shopAllBrandLink"]')
            if brand_el:
                raw_brand = brand_el.get_text(strip=True)
                # Remove "Shop all" prefix that Target prepends ("Shop allA New Day")
                product.brand = re.sub(r'^Shop\s+all\s*', '', raw_brand, flags=re.I).strip()

            # ── Breadcrumb / category ────────────────────────────────────────
            bc_els = soup.select('[data-test="@web/Breadcrumbs/BreadcrumbLink"]')
            if bc_els:
                crumbs = [el.get_text(strip=True) for el in bc_els]
                product.breadcrumb = " > ".join(crumbs)
                # Leaf crumb is the most specific category (skip "Target" root)
                product.leaf_category = crumbs[-1] if len(crumbs) > 1 else ""

            # ── Current price ────────────────────────────────────────────────
            price_el = soup.select_one('[data-test="product-price"]')
            if price_el:
                product.price = self._parse_price(price_el.get_text(strip=True))

            # ── Original / sale price ────────────────────────────────────────
            # Target shows original price in a strikethrough when on sale
            orig_candidates = [
                soup.select_one('[data-test="product-regular-price"]'),
                soup.select_one('[data-test="regular-price"]'),
            ]
            for cand in orig_candidates:
                if cand:
                    v = self._parse_price(cand.get_text(strip=True))
                    if v > 0:
                        product.original_price = v
                        break
            # Also look for displayed strikethrough price in the same price block.
            # Target uses data-test="strikethroughPriceMessage" (caught by *="strikethrough")
            # as well as plain <s>/<del> elements; scope to the price block parent so
            # we never accidentally pick up related-product carousels further down the page.
            if not product.original_price:
                price_block = soup.select_one('[data-test="product-price"]')
                if price_block:
                    parent = price_block.find_parent() or price_block
                    strike = parent.select_one(
                        "s, del, "
                        "[data-test*='strikethrough'], "
                        "[data-test*='was-price'], "
                        "[data-test*='regular-price'], "
                        "[class*=strike], [class*=Strike]"
                    )
                    if strike:
                        v = self._parse_price(strike.get_text(strip=True))
                        if v > 0:
                            product.original_price = v
            # Sale flag
            if product.original_price > 0 and product.price < product.original_price:
                product.is_on_sale = True
                if product.original_price:
                    product.discount_percent = round(
                        (product.original_price - product.price) / product.original_price * 100, 1
                    )

            # ── Rating & review count ────────────────────────────────────────
            # Primary: data-test="ratings" has full text like
            # "4.6 out of 5 stars with 31 ratings"
            for rating_sel in [
                '[data-test="ratings"]',
                '[data-test*="rating"]',
                '[aria-label*="out of 5"]',
            ]:
                rating_el = soup.select_one(rating_sel)
                if rating_el:
                    rt = rating_el.get_text(strip=True)
                    rm = re.search(r'([\d.]+)\s+out of\s+5', rt, re.I)
                    if rm:
                        try:
                            product.rating = float(rm.group(1))
                        except ValueError:
                            pass
                    rv = re.search(r'with\s+([\d,]+)\s+(?:rating|review)', rt, re.I)
                    if rv:
                        try:
                            product.review_count = int(rv.group(1).replace(',', ''))
                        except ValueError:
                            pass
                    if product.rating:
                        break

            # ── Feature bullets (highlights accordion) ───────────────────────
            highlight_sel = (
                soup.select_one('[data-test="item-details-highlights"]') or
                soup.select_one('[data-test*="highlights"]') or
                soup.select_one('[class*="highlights"]')
            )
            if highlight_sel:
                bullets = [li.get_text(strip=True) for li in highlight_sel.select("li") if li.get_text(strip=True)]
                product.feature_bullets = bullets

            # ── Images ──────────────────────────────────────────────────────
            seen_imgs: set = set()
            # Strategy A: <img> tags pointing to scene7     
            for img in soup.select('img[src*="scene7.com"]'):
                src = img.get("src", "")
                if src and "scene7.com" in src:
                    clean = re.sub(r'\?.*', '?wid=1200&hei=1200&qlt=85', src)
                    if clean not in seen_imgs:
                        seen_imgs.add(clean)
                        product.images.append(clean)
            # Strategy B: <source srcset> elements
            for src_el in soup.select('source[srcset*="scene7.com"]'):
                for part in src_el.get("srcset", "").split(","):
                    url_part = part.strip().split()[0]
                    if "scene7.com" in url_part:
                        clean = re.sub(r'\?.*', '?wid=1200&hei=1200&qlt=85', url_part)
                        if clean not in seen_imgs:
                            seen_imgs.add(clean)
                            product.images.append(clean)

            # ── Fit & style (PdpHighlightsSection) ───────────────────────────
            fit_and_style_bullets: List[str] = []
            fit_section = soup.select_one('#PdpHighlightsSection, [data-test="@web/ProductDetailPageHighlights"]')
            if fit_section:
                for h2 in fit_section.find_all('h2'):
                    h2_text = h2.get_text(strip=True).lower()
                    if 'fit' in h2_text or 'style' in h2_text:
                        ul = h2.find_next_sibling('ul')
                        if ul:
                            for li in ul.select('li'):
                                txt = li.get_text(strip=True)
                                if txt:
                                    fit_and_style_bullets.append(txt)
                        break
            fit_and_style_text = "; ".join(fit_and_style_bullets) if fit_and_style_bullets else ""

            # ── Description ──────────────────────────────────────────────────
            # When description not available, use Fit & style. When both available, combine both.
            desc = soup.select_one('[data-test="item-details-description"]')
            desc_text = ""
            if desc:
                desc_text = " ".join(desc.stripped_strings)[:1500].strip()
            if desc_text and fit_and_style_text:
                product.description = f"{desc_text} {fit_and_style_text}".strip()
            elif fit_and_style_text:
                product.description = fit_and_style_text
            elif desc_text:
                product.description = desc_text

            # ── Specs ────────────────────────────────────────────────────────
            # Primary: parse the rendered text right from BeautifulSoup
            spec_el = soup.select_one('[data-test="item-details-specifications"]')
            parsed_specs = self._parse_specs_text(spec_el)

            # Fallback JS extraction for structured dt/dd or table layouts
            if not parsed_specs:
                raw_specs = await page.evaluate("""
                () => {
                    const results = {};
                    let panel = document.querySelector('[data-test="item-details-specifications"]');
                    if (!panel) return results;

                    // dl/dt+dd
                    panel.querySelectorAll('dt').forEach(dt => {
                        const dd = dt.nextElementSibling;
                        if (dd && dd.tagName === 'DD') {
                            const k = dt.innerText.trim().replace(/:$/, '');
                            const v = dd.innerText.trim();
                            if (k && v) results[k] = v;
                        }
                    });
                    // tr/td table
                    if (Object.keys(results).length === 0) {
                        panel.querySelectorAll('tr').forEach(tr => {
                            const cells = tr.querySelectorAll('td, th');
                            if (cells.length >= 2) {
                                const k = cells[0].innerText.trim().replace(/:$/, '');
                                const v = cells[1].innerText.trim();
                                if (k && v) results[k] = v;
                            }
                        });
                    }
                    // Sibling-div pattern
                    if (Object.keys(results).length === 0) {
                        panel.querySelectorAll('div').forEach(row => {
                            const children = Array.from(row.children).filter(
                                c => c.children.length === 0 && c.innerText?.trim()
                            );
                            if (children.length === 2) {
                                const k = children[0].innerText.trim().replace(/:$/, '');
                                const v = children[1].innerText.trim();
                                if (k && v) results[k] = v;
                            }
                        });
                    }
                    return results;
                }
                """)
                parsed_specs = raw_specs or {}

            self._apply_specs(product, parsed_specs)

            # ── Color swatches ───────────────────────────────────────────────
            color_set: set = set()
            # VariationComponent: <a aria-label="Color, Beige">, <a aria-label="Color, Light Pink">
            for swatch in soup.select(
                'a[aria-label*="Color,"], a[aria-label*="Color, "], '
                'button[aria-label*="color"], '
                '[data-test*="colorSwatch"] button, [data-test*="colorSwatch"] a, '
                '[data-test="@web/VariationComponent"] a[aria-label*="Color"], '
                '[class*="SwatchChip"] button, [class*="ndsChip"] a, '
                'button[data-variant-id]'
            ):
                label = swatch.get("aria-label", "").strip()
                if label:
                    # "Color, Beige" or "Color, Light Pink, selected" -> "Beige", "Light Pink"
                    color_name = re.split(r',\s*select', label, flags=re.I)[0].strip()
                    color_name = re.sub(r'^Color,\s*', '', color_name, flags=re.I).strip()
                    color_name = re.sub(r',\s*selected\s*$', '', color_name, flags=re.I).strip()
                    if color_name and len(color_name) < 50:
                        color_set.add(color_name)
            # Fallback: header shows selected color, e.g. "Color Light Pink" in headerWrapper
            if not color_set:
                var_comp = soup.select_one('[data-test="@web/VariationComponent"]')
                if var_comp:
                    header_div = var_comp.select_one('[class*="headerWrapper"]')
                    if header_div:
                        full_text = header_div.get_text(strip=True)
                        color_txt = re.sub(r'^Color\s*', '', full_text, flags=re.I).strip()
                        if color_txt and len(color_txt) < 50 and color_txt.lower() not in ('color', 'size'):
                            color_set.add(color_txt)
            if color_set:
                product.colors = sorted(color_set)

            # ── is_new badge ─────────────────────────────────────────────────
            if not product.is_new:
                new_badge = soup.select_one(
                    '[data-test*="new-badge"], [class*="newBadge"], [class*="NewBadge"]'
                )
                product.is_new = new_badge is not None

            # ── In-stock / availability ──────────────────────────────────────
            add_to_cart = soup.select_one('[data-test*="AddToCart"], [data-test*="fulfillment"]')
            if add_to_cart:
                atc_text = add_to_cart.get_text(strip=True).lower()
                product.in_stock = "out of stock" not in atc_text and "unavailable" not in atc_text
            
            logger.info(f"  ✓ Successfully enriched {product.tcin}")

        except Exception as e:
            logger.warning(f"Detail enrichment failed for {product.tcin} ({product.title}): {type(e).__name__}: {str(e)[:200]}")
            import traceback
            logger.debug(f"Full traceback:\n{traceback.format_exc()}")

    def _parse_price(self, s: str) -> float:
        if not s:
            return 0.0
        try:
            cleaned = re.sub(r'[^\d.]', '', s)
            return float(cleaned) if cleaned else 0.0
        except:
            return 0.0

    # ─── Main flow ───────────────────────────────────────────────────────────

    async def run(self):
        if not self.skip_dedup:
            self._load_metadata()
        else:
            logger.info("Dedup disabled (--fresh) – will crawl all products")
        max_proxy_attempts = len(self.proxy_pool) if self.proxy_pool else 1

        for attempt in range(max_proxy_attempts):
            if attempt > 0:
                self.products = []
                logger.info(f"Retrying with proxy {self._proxy_index + 1}/{len(self.proxy_pool)}")

            try:
                context, listing_page = await self._init_browser()
            except Exception as e:
                if self.proxy_pool and self._is_proxy_error(e) and self._rotate_proxy():
                    logger.warning(f"Proxy failed, rotating to next: {str(e)[:120]}")
                    continue
                raise

            try:
                # ═══════════════════════════════════════════════════════════════
                # PHASE 1: Collect all product URLs from listing pages
                # ═══════════════════════════════════════════════════════════════
                logger.info("=" * 70)
                logger.info("PHASE 1: Scraping listing pages to collect product URLs")
                logger.info("=" * 70)

                if self.concurrent_listing_pages > 1:
                    # Parallel mode: open N listing pages simultaneously
                    await listing_page.close()  # Free the initial page; we create N new ones
                    num_pages = self.concurrent_listing_pages
                    urls = self._build_listing_urls(num_pages)
                    logger.info(f"Opening {num_pages} listing pages simultaneously → {urls[0]} ... Nao={num_pages * self.ITEMS_PER_PAGE - self.ITEMS_PER_PAGE}")

                    async def scrape_one_listing(ctx: BrowserContext, idx: int, u: str) -> List[Product]:
                        page = await ctx.new_page()
                        try:
                            products = await self._scrape_listing_page(page, u)
                            logger.info(f"  Page {idx + 1}/{num_pages} ({u}) → {len(products)} products")
                            return products
                        finally:
                            await page.close()

                    tasks = [scrape_one_listing(context, i, u) for i, u in enumerate(urls)]
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    seen_tcins: set = set()
                    any_proxy_error = False
                    for i, r in enumerate(results):
                        if isinstance(r, Exception):
                            logger.warning(f"  Page {i + 1} failed: {type(r).__name__}: {str(r)[:150]}")
                            if self._is_proxy_error(r):
                                any_proxy_error = True
                            continue
                        for p in r:
                            if p.tcin and p.tcin not in seen_tcins:
                                seen_tcins.add(p.tcin)
                                self.products.append(p)
                                if self.max_products and len(self.products) >= self.max_products:
                                    break
                        if self.max_products and len(self.products) >= self.max_products:
                            break

                    if self.max_products:
                        self.products = self.products[:self.max_products]
                    # Dedup: skip already-fetched products
                    if not self.skip_dedup:
                        before_dedup = len(self.products)
                        self.products = [p for p in self.products if p.tcin and p.tcin not in self._fetched_tcins]
                        skipped = before_dedup - len(self.products)
                        if skipped:
                            logger.info(f"Dedup: skipped {skipped} already-fetched products")
                    logger.info(f"Phase 1 complete. Collected {len(self.products)} NEW products from {num_pages} parallel pages")
                    if len(self.products) == 0 and any_proxy_error and self.proxy_pool and self._rotate_proxy():
                        logger.warning("All parallel pages failed with proxy errors, rotating and retrying")
                        await self._close_browser()
                        continue
                else:
                    # Sequential mode (original behavior)
                    url = self.base_url
                    page_num = 1

                    while url and (not self.max_products or len(self.products) < self.max_products):
                        logger.info(f"Listing Page {page_num}  ──  {url}")

                        new_products = await self._scrape_listing_page(listing_page, url)
                        self.products.extend(new_products)

                        if len(new_products) == 0:
                            logger.warning("No products found → likely end of results or block")
                            break

                        if self.max_products and len(self.products) >= self.max_products:
                            self.products = self.products[:self.max_products]
                            logger.info(f"Reached max_products limit ({self.max_products})")
                            break

                        next_url = await self._get_next_page_url(listing_page)
                        if not next_url:
                            logger.info("No more pages")
                            break

                        url = next_url
                        page_num += 1
                        await self._delay()

                    # Dedup: skip already-fetched products
                    if not self.skip_dedup:
                        before_dedup = len(self.products)
                        self.products = [p for p in self.products if p.tcin and p.tcin not in self._fetched_tcins]
                        skipped = before_dedup - len(self.products)
                        if skipped:
                            logger.info(f"Dedup: skipped {skipped} already-fetched products")
                    logger.info(f"Phase 1 complete. Collected {len(self.products)} NEW products from {page_num} page(s)")

                # ═══════════════════════════════════════════════════════════════
                # PHASE 2: Visit product detail pages (concurrent when --concurrent-pdp > 1)
                # ═══════════════════════════════════════════════════════════════
                if self.get_details and self.products:
                    logger.info("=" * 70)
                    logger.info(f"PHASE 2: Enriching {len(self.products)} products (concurrent_pdp={self.concurrent_pdp})")
                    logger.info("=" * 70)

                    products_with_url = [p for p in self.products if p.url]
                    if len(products_with_url) < len(self.products):
                        logger.warning(f"Skipped {len(self.products) - len(products_with_url)} products without URL")

                    for batch_start in range(0, len(products_with_url), self.concurrent_pdp):
                        batch = products_with_url[batch_start : batch_start + self.concurrent_pdp]
                        batch_num = batch_start // self.concurrent_pdp + 1
                        total_batches = (len(products_with_url) + self.concurrent_pdp - 1) // self.concurrent_pdp

                        async def fetch_one_pdp(prod: Product) -> None:
                            page = await context.new_page()
                            try:
                                await self._enrich_with_details(page, prod)
                            except Exception as e:
                                logger.warning(f"Failed to enrich {prod.tcin}: {type(e).__name__}: {str(e)[:150]}")
                            finally:
                                await page.close()

                        tasks = [fetch_one_pdp(p) for p in batch]
                        await asyncio.gather(*tasks)

                        logger.info(f"  Batch {batch_num}/{total_batches} done ({len(batch)} PDPs)")
                        if batch_start + len(batch) < len(products_with_url):
                            await self._delay()

                logger.info("=" * 70)
                logger.info(f"COMPLETE: Total products collected: {len(self.products)}")
                logger.info("=" * 70)

                self._save_results()
                break  # Success – exit retry loop
            except Exception as e:
                if self.proxy_pool and self._is_proxy_error(e) and self._rotate_proxy():
                    logger.warning(f"Scrape failed, rotating proxy: {str(e)[:120]}")
                    await self._close_browser()
                    continue
                raise
            finally:
                await self._close_browser()

    async def _get_next_page_url(self, page: Page) -> Optional[str]:
        """Find and click the next page button, return new URL if successful."""
        try:
            # Scroll to bottom first to make pagination visible
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(0.3)
            
            # Wait for pagination container
            try:
                await page.wait_for_selector('[data-test="listing-page-pagination"]', timeout=5000)
            except:
                logger.debug("Pagination container not found")
                return None
            
            # Check if next button exists and is clickable
            next_btn = page.locator('button[data-test="next"]')
            count = await next_btn.count()
            logger.debug(f"Found {count} next button(s)")
            
            if count == 0:
                logger.debug("No next button found - likely last page")
                return None
            
            # Check if button is disabled by checking for disabled attribute or aria-disabled
            is_disabled = await page.evaluate("""
                () => {
                    const btn = document.querySelector('button[data-test="next"]');
                    if (!btn) return true;
                    return btn.disabled || btn.getAttribute('aria-disabled') === 'true' || 
                           btn.classList.contains('disabled');
                }
            """)
            
            if is_disabled:
                logger.debug("Next button is disabled - reached last page")
                return None
            
            # Check current page info
            page_info = await page.evaluate("""
                () => {
                    const pageText = document.querySelector('[data-test="select"] span')?.textContent || '';
                    return pageText;
                }
            """)
            logger.debug(f"Current pagination: {page_info}")

            current_url = page.url
            logger.debug(f"Clicking next button (current URL: {current_url})")
            
            # Scroll the button into view and click
            await next_btn.first.scroll_into_view_if_needed()
            await asyncio.sleep(0.3)
            await next_btn.first.click()
            
            # Wait for navigation - check URL or product grid changes
            await asyncio.sleep(2.5)
            
            # Wait for either URL change or product grid to reload
            for attempt in range(20):
                new_url = page.url
                if new_url != current_url:
                    logger.debug(f"✓ URL changed to: {new_url}")
                    # Wait for products to load on new page
                    try:
                        await page.wait_for_selector('[data-test="@web/ProductCard/title"]', timeout=8000)
                    except:
                        pass
                    return new_url
                await asyncio.sleep(0.5)

            logger.warning("Next button clicked but URL didn't change after 10s - pagination may have failed")
            return None
            
        except Exception as e:
            logger.warning(f"Pagination error: {type(e).__name__}: {str(e)[:150]}")
            return None

    def _save_results(self):
        import csv
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        rows = [p.to_dict() for p in self.products]

        if not rows:
            logger.warning("No products to save.")
            return

        # JSON
        path_json = self.output_dir / f"target_handbags_{ts}.json"
        with open(path_json, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved JSON:  {path_json}")

        # JSONL
        path_jsonl = self.output_dir / f"target_handbags_{ts}.jsonl"
        with open(path_jsonl, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info(f"Saved JSONL: {path_jsonl}")

        # CSV
        path_csv = self.output_dir / f"target_handbags_{ts}.csv"
        fieldnames = list(rows[0].keys())
        with open(path_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Saved CSV:   {path_csv}")

        # Update metadata so these products are not re-crawled next run
        if not self.skip_dedup:
            new_tcins = [p.tcin for p in self.products if p.tcin]
            if new_tcins:
                self._save_metadata(new_tcins)


# ────────────────────────────────────────────────
# CLI entry point
# ────────────────────────────────────────────────

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Target Handbags Scraper (simplified)")
    parser.add_argument("--max-products", type=int, default=None, help="Stop after N products")
    parser.add_argument("--details", action="store_true", help="Also scrape detail pages")
    parser.add_argument("--output-dir", default="./data", help="Where to save results")
    parser.add_argument("--fresh", action="store_true", help="Disable dedup – crawl all products (ignore metadata)")
    parser.add_argument("--concurrent-pages", type=int, default=1, help="Open N listing pages simultaneously (default 1 = sequential)")
    parser.add_argument("--concurrent-pdp", type=int, default=1, help="Fetch N product detail pages simultaneously when --details (default 1)")
    parser.add_argument("--headless", action="store_true", default=True, help="Run headless (default)")
    parser.add_argument("--headed", action="store_true", help="Show browser window")
    parser.add_argument("--devtools", action="store_true", help="Open Chromium DevTools (forces headed)")
    parser.add_argument("--slow-mo", type=int, default=0, help="Slow down Playwright actions in ms")
    parser.add_argument("--proxy", default=None, help="Single proxy, e.g. 50.203.147.152:80 or http://user:pass@host:port")
    parser.add_argument("--proxy-file", default=None, help="Load proxies from file (one per line, ip:port). Rotates on failure.")
    parser.add_argument("--verbose", action="store_true", help="More logging")
    return parser.parse_args()


async def main():
    args = parse_args()

    scraper = TargetHandbagsScraper(
        max_products=args.max_products,
        get_details=args.details,
        output_dir=args.output_dir,
        skip_dedup=args.fresh,
        concurrent_listing_pages=args.concurrent_pages,
        concurrent_pdp=args.concurrent_pdp,
        headless=not args.headed,
        devtools=args.devtools,
        slow_mo=args.slow_mo,
        verbose=args.verbose,
        proxy=args.proxy,
        proxy_file=args.proxy_file,
    )

    await scraper.run()


if __name__ == "__main__":
    asyncio.run(main())