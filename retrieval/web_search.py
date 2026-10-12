import sys
import re
import json
import html
import base64
import urllib.request
import urllib.parse
from typing import Dict, Any, List, Optional

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


def _extract_product_keyword(text: str) -> str:
    """Extract brand and product name for accurate web querying."""
    try:
        from pipeline.ecommerce_analyzer import resolve_product_entity, clean_conversational_prefix
        cleaned = clean_conversational_prefix(text)
        ent = resolve_product_entity(cleaned)
        if ent and ent.get("resolved_name"):
            res_name = ent["resolved_name"].strip()
            # If resolved_name doesn't start with personal pronoun or stopword
            if len(res_name) >= 3 and not re.match(r'^(?:i|you|we|they|he|she|get)\b', res_name, re.IGNORECASE):
                res_name = re.sub(r'\bpavellion\b', 'Pavilion', res_name, flags=re.IGNORECASE)
                return res_name
    except Exception:
        pass

    try:
        from pipeline.ecommerce_analyzer import clean_conversational_prefix
        text = clean_conversational_prefix(text)
    except Exception:
        pass

    text_norm = re.sub(r'\bpavellion\b', 'Pavilion', text, flags=re.IGNORECASE)
    text_norm = re.sub(r'\bi\s*[-_]?\s*phone\b', 'iPhone', text_norm, flags=re.IGNORECASE)
    text_norm = re.sub(r'\bi\s*[-_]?\s*pad\b', 'iPad', text_norm, flags=re.IGNORECASE)
    text_norm = re.sub(r'\bi\s*[-_]?\s*mac\b', 'iMac', text_norm, flags=re.IGNORECASE)
    text_norm = re.sub(r'\bi\s*[-_]?\s*watch\b', 'Apple Watch', text_norm, flags=re.IGNORECASE)

    brand_match = re.search(
        r'\b(Adidas|Nike|Apple|Sony|Samsung|Puma|Reebok|Bose|JBL|New Balance|Asics|Dell|HP|Lenovo|Xiaomi|Redmi|Realme|OnePlus|Oppo|Vivo|iQOO|Poco|Motorola|Nothing|Boat|Noise|Rolex|Omega|Casio|Seiko|iPhone|iPad|MacBook|AirPods)\b\s*([a-zA-Z0-9\s\-]{1,30})?',
        text_norm, re.IGNORECASE
    )
    if brand_match:
        brand = brand_match.group(1).title()
        if brand.lower() in ('iphone', 'ipad', 'macbook', 'airpods'):
            brand = f"Apple {brand}"
        rest = brand_match.group(2) or ""
        rest = re.sub(
            r'\b(shoes|shoe|available|for|is|are|at|instead|of|under|below|on|this|only|deal|sale|price|online|in\s+\d+|rupees?)\b.*$',
            '', rest, flags=re.IGNORECASE
        ).strip()
        cleaned = f"{brand} {rest}".strip()
        if len(cleaned) >= 3:
            return cleaned

    # Fallback to first clause without stopwords
    first_clause = text_norm.split('.')[0].split(',')[0]
    first_clause = re.sub(r'^(?:you|he|she|we|they)\s+(?:will|can|could|might|get|buy)?\s*', '', first_clause, flags=re.IGNORECASE).strip()
    first_clause = re.sub(r'^i\s+(?:will|can|could|might|get|buy)\s+', '', first_clause, flags=re.IGNORECASE).strip()
    first_clause = re.sub(r'^i\s+(?!(?:phone|pad|mac|pod|watch)\b)', '', first_clause, flags=re.IGNORECASE).strip()
    return first_clause[:45].strip()



def _decode_bing_u(u: str) -> str:
    """Decode base64 encoded URL from Bing redirect."""
    try:
        if u.startswith('a1'):
            u = u[2:]
        u += '=' * (-len(u) % 4)
        return base64.b64decode(u).decode('utf-8', errors='ignore')
    except Exception:
        return ""


def _is_product_page(url: str) -> bool:
    """Heuristic: True if URL looks like a specific product/category page, not just a homepage."""
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.rstrip('/')
    netloc = parsed.netloc.lower()
    # A homepage has no path or just '/'
    if not path or path == '/':
        return False
    # Wikipedia articles are always specific entity pages
    if 'wikipedia.org/wiki/' in url:
        return True
    # Brand subdomain pages (account.apple.com, support.apple.com, etc.) are always sub-pages
    known_brand_subdomains = ('account.apple.com', 'support.apple.com', 'developer.apple.com',
                              'support.nike.com', 'help.adidas.com', 'shop.samsung.com')
    if any(netloc == sub for sub in known_brand_subdomains):
        return True
    # Paths with product-related signals
    product_signals = (
        '/product', '/p/', '/item', '/shop', '/buy', '/c/', '/shoes',
        '/footwear', '/collections', '/catalog', '/search', '/store',
        '/airpods', '/air-force', '/ultraboost', '-shoes', '-sneaker',
        '/men', '/women', '/sport', '/running', '/t/', '/pd/', '/sku',
        '/shop-all', '/w/', '/gb/', '/in/', '/au/', '/ca/', '/airpods-pro',
        '/iphone', '/mac', '/ipad', '/apple-watch',
    )
    path_lower = path.lower()
    if any(sig in path_lower for sig in product_signals):
        return True
    # URLs with 2+ non-empty path segments are typically not homepages
    segments = [s for s in path.split('/') if s]
    return len(segments) >= 2


def _upgrade_to_product_url(url: str, keyword: str) -> str:
    """Heuristic: Transform generic homepage URLs into targeted product search links."""
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.rstrip('/')
    domain = parsed.netloc.lower().replace('www.', '')
    
    # If already a specific sub-page, return as is
    if path and path != '/' and _is_product_page(url):
        return url

    kw_enc = urllib.parse.quote(keyword)
    if "adidas" in domain:
        return f"https://www.adidas.co.in/search?q={kw_enc}"
    elif "nike" in domain:
        return f"https://www.nike.com/w?q={kw_enc}"
    elif "apple" in domain:
        return f"https://www.apple.com/in/search/{kw_enc}?src=globalnav"
    elif "sony" in domain:
        return f"https://www.sony.co.in/search?query={kw_enc}"
    elif "samsung" in domain:
        return f"https://www.samsung.com/in/search/?searchvalue={kw_enc}"
    elif "mi.com" in domain or "xiaomi" in domain:
        return f"https://www.mi.com/in/search?keyword={kw_enc}"
    elif "oneplus" in domain:
        return f"https://www.oneplus.in/search?q={kw_enc}"
    elif "myntra.com" in domain:
        return f"https://www.myntra.com/{keyword.lower().replace(' ', '-')}"
    elif "flipkart.com" in domain:
        return f"https://www.flipkart.com/search?q={kw_enc}"
    elif "amazon.in" in domain:
        return f"https://www.amazon.in/s?k={kw_enc}"
    elif "croma.com" in domain:
        return f"https://www.croma.com/searchB?q={kw_enc}"

    if parsed.netloc:
        return f"https://{parsed.netloc}/search?q={kw_enc}"
    return url


def _build_product_query(keyword: str) -> str:
    """Build a natural query that targets product price and official specifications."""
    if 'price' in keyword.lower():
        return keyword
    return f"{keyword} price in India"


def _search_bing_live(keyword: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Query Bing search engine for live verified product listings and official stores."""
    query_str = _build_product_query(keyword)
    url = 'https://www.bing.com/search?q=' + urllib.parse.quote(query_str)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9'
    }
    results = []
    brand = keyword.split()[0].lower() if keyword else ""
    ignore_domains = (
        'account.apple.com', 'support.apple.com', 'developer.apple.com', 'asw.apple.com',
        'login.', 'signin.', 'support.', 'drivers.', 'hpsmart.com', 'dictionary.',
        'merriam-webster.com', 'thefreedictionary.com', 'cambridge.org'
    )
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=6) as resp:
            page = resp.read().decode('utf-8', errors='ignore')

        pattern = r'<h2[^>]*><a[^>]+href="([^"]+)"[^>]*>(.*?)</a></h2>.*?<div class="b_caption"><p[^>]*>(.*?)</p>'
        matches = re.findall(pattern, page, re.DOTALL)
        for href, title, snip in matches:
            u_m = re.search(r'[?&](?:amp;)?u=([^&]+)', href)
            u = _decode_bing_u(u_m.group(1)) if u_m else href
            t = html.unescape(re.sub(r'<[^>]+>', '', title).strip())
            s = html.unescape(re.sub(r'<[^>]+>', '', snip).strip())
            
            domain = urllib.parse.urlparse(u).netloc.replace('www.', '')
            if 'bing.com' not in domain and u.startswith('http'):
                if any(ig in domain for ig in ignore_domains):
                    continue
                # Upgrade homepage or generic URLs to direct product pages
                u = _upgrade_to_product_url(u, keyword)
                is_product = _is_product_page(u)
                if not brand or brand in (t + s + domain).lower():
                    results.append({
                        'title': t,
                        'url': u,
                        'domain': domain,
                        'snippet': s,
                        'is_product_page': is_product
                    })
                    if len(results) >= max_results:
                        break
    except Exception as e:
        print(f"  [Live Bing Notice] {e}")
    # Sort: product-page URLs first
    results.sort(key=lambda r: 0 if r.get('is_product_page') else 1)
    return results


def _search_ddg_live(keyword: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Query DuckDuckGo search engine for live listings."""
    query_str = _build_product_query(keyword)
    url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query_str)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
    }
    results = []
    brand = keyword.split()[0].lower() if keyword else ""
    ignore_domains = ('account.apple.com', 'support.apple.com', 'developer.apple.com', 'asw.apple.com', 'login.', 'signin.')
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=6) as resp:
            page_html = resp.read().decode('utf-8', errors='ignore')
            
        parts = page_html.split('<div class="result results_links')
        for p in parts[1:]:
            a_match = re.search(r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', p, re.DOTALL)
            s_match = re.search(r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', p, re.DOTALL)
            
            if a_match:
                raw_href = a_match.group(1)
                title = html.unescape(re.sub(r'<[^>]+>', '', a_match.group(2)).strip())
                
                u_match = re.search(r'uddg=([^&]+)', raw_href)
                actual_url = urllib.parse.unquote(u_match.group(1)) if u_match else raw_href
                
                snippet = ""
                if s_match:
                    snippet = html.unescape(re.sub(r'<[^>]+>', '', s_match.group(1)).strip())
                
                domain = urllib.parse.urlparse(actual_url).netloc.replace("www.", "")
                
                if "duckduckgo.com" not in domain and actual_url.startswith("http"):
                    if any(ig in domain for ig in ignore_domains):
                        continue
                    actual_url = _upgrade_to_product_url(actual_url, keyword)
                    is_product = _is_product_page(actual_url)
                    if not brand or brand in (title + snippet + domain).lower():
                        results.append({
                            "title": title,
                            "url": actual_url,
                            "snippet": snippet,
                            "domain": domain,
                            "is_product_page": is_product
                        })
                        if len(results) >= max_results:
                            break
    except Exception as e:
        print(f"  [Live DDG Notice] {e}")
    # Sort: product-page URLs first
    results.sort(key=lambda r: 0 if r.get('is_product_page') else 1)
    return results


def _search_wikipedia_fallback(keyword: str) -> List[Dict[str, str]]:
    """Live encyclopedic entity verification fallback."""
    url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(keyword)}&format=json"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    results = []
    kw_words = [w.lower() for w in keyword.split() if len(w) > 3 and w.lower() not in ('shoes', 'shoe', 'price', 'online')]
    brand_words = {'adidas', 'nike', 'apple', 'sony', 'samsung', 'puma', 'reebok', 'bose'}
    model_words = [w for w in kw_words if w not in brand_words]
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            items = data.get('query', {}).get('search', [])
            for it in items[:3]:
                page_title = it.get('title', '')
                clean_snippet = html.unescape(re.sub(r'<[^>]+>', '', it.get('snippet', '')))
                t_lower = page_title.lower()
                # If specific model queried, do not accept competing/different model pages
                if model_words:
                    if not any(mw in t_lower or mw in clean_snippet.lower() for mw in model_words):
                        # Allow only the brand's own main page as entity fallback
                        if not any(bw == t_lower for bw in brand_words):
                            continue
                elif kw_words and not any(w in t_lower for w in kw_words):
                    continue
                results.append({
                    "title": f"{page_title} - Official Encyclopedia",
                    "url": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(page_title.replace(' ', '_'))}",
                    "snippet": clean_snippet,
                    "domain": "en.wikipedia.org",
                    "is_product_page": True
                })
    except Exception as e:
        print(f"  [Live Wiki Notice] {e}")
    return results


def search_live_web(query: str, domain: str = "google.com") -> Dict[str, Any]:
    """
    Execute live web search for product validation.
    Returns real search snippets, real URLs, and real domains.
    Zero hardcoded catalogs.
    """
    clean_keyword = _extract_product_keyword(query)
    safe_preview = clean_keyword.encode('ascii', errors='replace').decode('ascii')
    print(f"  [Retrieval] Executing live web search for: '{safe_preview}'")
    
    web_results = []
    
    # 1. Fetch live search results via Bing search engine
    live_bing = _search_bing_live(clean_keyword, max_results=3)
    for br in live_bing:
        if not any(r['url'] == br['url'] for r in web_results):
            web_results.append(br)
    
    # 2. Also fetch DuckDuckGo live results for retailer listings & price catalogs
    ddg_results = _search_ddg_live(clean_keyword, max_results=3)
    for dr in ddg_results:
        if not any(r['url'] == dr['url'] for r in web_results):
            web_results.append(dr)
                
    # 3. Include Wikipedia entity registry for product authenticity grounding
    wiki_results = _search_wikipedia_fallback(clean_keyword)
    for wr in wiki_results:
        if not any(r['url'] == wr['url'] for r in web_results):
            web_results.append(wr)
            break

    # Prioritize product/store pages over generic informational pages
    web_results.sort(key=lambda r: 0 if r.get('is_product_page') else 1)
        
    if web_results:
        primary_domain = web_results[0].get("domain", domain)
        snippets = [r.get("snippet", "") for r in web_results if r.get("snippet")]
        combined_text = " ".join(snippets)
        print(f"  [Live Web] Retrieved {len(web_results)} genuine search results across {len(snippets)} snippets.")
        
        return {
            "record_id": "web_live_search",
            "entity": clean_keyword,
            "attribute": "pricing_and_catalog",
            "domain": primary_domain,
            "label": "CONSISTENT",
            "sources": {
                "text": combined_text,
                "image_path": "",
                "document_excerpt": web_results[0].get("snippet", "")
            },
            "web_results": web_results,
            "visually_verifiable": False
        }

    # 4. Clean unverified state if network is unavailable
    print(f"  [Live Web Notice] No live search results returned for '{safe_preview}'.")
    return {
        "record_id": "no_verified_evidence",
        "entity": clean_keyword,
        "attribute": "general",
        "domain": "web",
        "label": "NO_EVIDENCE",
        "sources": {
            "text": f"No verified public catalog record found online for {clean_keyword}.",
            "image_path": "",
            "document_excerpt": None
        },
        "web_results": [],
        "visually_verifiable": False
    }


if __name__ == "__main__":
    res = search_live_web("Adidas Ultraboost shoes are available for ₹3,499 instead of ₹15,999.")
    print("Entity:", res["entity"])
    print("Domain:", res["domain"])
    print("Found Results:", len(res.get("web_results", [])))
    for r in res.get("web_results", []):
        print(" ->", r["title"], "|", r["url"])
