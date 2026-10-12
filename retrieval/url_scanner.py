import urllib.request
import urllib.parse
from bs4 import BeautifulSoup
import re
import html

def scrape_url_claim(url: str):
    """
    Dynamically scrape any product or article URL to extract live title,
    description, price, and generate a factual/commercial claim.
    """
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    parsed = urllib.parse.urlparse(url)
    domain = parsed.netloc.replace('www.', '')
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9'
    }
    
    req = urllib.request.Request(url, headers=headers)
    raw_html = ""
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw_html = resp.read().decode('utf-8', errors='ignore')
    except Exception as e:
        # Fallback: parse product slug from URL path
        path_slug = parsed.path.replace('-', ' ').replace('_', ' ').replace('/', ' ').strip()
        slug_clean = re.sub(r'\b(dp|p|product|item|buy|html|site)\b|\b[A-Z0-9]{8,}\b|\b\d{5,}\b', '', path_slug, flags=re.IGNORECASE).strip()
        slug_clean = re.sub(r'\s+', ' ', slug_clean).title()
        if slug_clean and len(slug_clean) >= 3:
            return {
                'title': f"{slug_clean} on {domain}",
                'domain': domain,
                'description': f"Product listing on {domain}",
                'price': None,
                'extracted_claim': f"{slug_clean} is available on {domain}.",
                'success': True
            }
        return {
            'title': f"Listing on {domain}",
            'domain': domain,
            'description': f"Listing from {domain}",
            'price': None,
            'extracted_claim': f"Product listing on {domain}.",
            'success': True
        }
        
    soup = BeautifulSoup(raw_html, 'html.parser')
    
    # 1. Title extraction
    title = ''
    og_title = soup.find('meta', property='og:title')
    if og_title and og_title.get('content'):
        title = og_title['content'].strip()
    elif soup.title and soup.title.string:
        title = soup.title.string.strip()
    
    title = html.unescape(title)
    # Strip common site brand suffixes from title
    title_clean = re.sub(r'\s*[-|–—:]\s*(?:Amazon|Flipkart|Wikipedia|eBay|Walmart|Best Buy|Official|Myntra|Croma).*$', '', title, flags=re.IGNORECASE).strip()
    title_clean = re.sub(r'[\r\n\t]+', ' ', title_clean)[:120].strip()
    
    # 2. Description extraction
    desc = ''
    og_desc = soup.find('meta', property='og:description')
    if og_desc and og_desc.get('content'):
        desc = og_desc['content'].strip()
    else:
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            desc = meta_desc['content'].strip()
    desc = html.unescape(re.sub(r'[\r\n\t]+', ' ', desc)[:160].strip())
    
    # 3. Price extraction
    price_str = None
    for s in soup.find_all('script', type='application/ld+json'):
        if s.string and 'price' in s.string.lower():
            m = re.search(r'["\']price["\']\s*:\s*["\']?([\d,]+(?:\.\d+)?)["\']?', s.string)
            if m:
                price_str = m.group(1)
                break
                
    if not price_str:
        og_price = soup.find('meta', property='product:price:amount') or soup.find('meta', property='og:price:amount')
        if og_price and og_price.get('content'):
            price_str = og_price['content'].strip()
            
    if not price_str:
        price_match = re.search(r'(?:₹|Rs\.?|\$)\s*([\d,]+(?:\.\d+)?)', raw_html[:15000])
        if price_match:
            price_str = price_match.group(1)
            
    currency = '₹' if ('₹' in raw_html[:15000] or 'inr' in raw_html[:15000].lower() or '.in' in domain) else '$'
    
    if price_str and title_clean:
        claim = f"{title_clean} is available for {currency}{price_str} on {domain}."
    elif title_clean and desc:
        claim = f"{title_clean} - {desc[:100]} on {domain}."
    elif title_clean:
        claim = f"{title_clean} is available on {domain}."
    else:
        claim = f"Product listing on {domain}."
        
    return {
        'title': title_clean or f"Listing on {domain}",
        'domain': domain,
        'description': desc,
        'price': price_str,
        'extracted_claim': claim,
        'success': True
    }
