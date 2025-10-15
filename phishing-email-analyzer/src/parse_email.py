import re
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import tldextract

def extract_urls_from_html(html):
    soup = BeautifulSoup(html or "", "html.parser")
    urls = []
    for a in soup.find_all('a', href=True):
        urls.append(a['href'])
    return urls

def extract_urls_from_text(text):
    return re.findall(r'https?://\S+', text or "")

def url_features(urls):
    feats = {
        "num_urls": len(urls),
        "num_ip_urls": 0,
        "num_shorteners": 0,
        "num_punycode": 0
    }
    shorteners = set(['bit.ly','tinyurl.com','t.co','goo.gl'])
    for u in urls:
        try:
            parsed = urlparse(u)
            host = parsed.hostname or ""
            if re.match(r'^\d+\.\d+\.\d+\.\d+$', host):
                feats['num_ip_urls'] += 1
            if host.startswith('xn--'):
                feats['num_punycode'] += 1
            t = tldextract.extract(host)
            domain = (t.domain + '.' + t.suffix) if t.suffix else t.domain
            if domain in shorteners:
                feats['num_shorteners'] += 1
        except Exception:
            continue
    return feats
