import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from .parse_email import extract_urls_from_html, extract_urls_from_text, url_features

def build_text_column(df):
    return df['subject'].fillna('') + " " + df['body_text'].fillna('')

def fit_vectorizer(texts, max_features=5000):
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1,2))
    vec.fit(texts)
    return vec

def add_url_numeric_features(df):
    num_urls = []
    num_ip = []
    num_short = []
    num_puny = []
    for _, row in df.iterrows():
        urls = extract_urls_from_html(row.get('body_html','')) + extract_urls_from_text(row.get('body_text',''))
        feats = url_features(urls)
        num_urls.append(feats['num_urls'])
        num_ip.append(feats['num_ip_urls'])
        num_short.append(feats['num_shorteners'])
        num_puny.append(feats['num_punycode'])
    df['num_urls'] = num_urls
    df['num_ip_urls'] = num_ip
    df['num_shorteners'] = num_short
    df['num_punycode'] = num_puny
    return df
