import joblib
from .feature_extraction import build_text_column, add_url_numeric_features
import pandas as pd
from scipy.sparse import hstack
from scipy import sparse

def predict_single(email_subject, email_body):
    # load model & vectorizer
    clf = joblib.load('models/model.pkl')
    vec = joblib.load('models/vectorizer.pkl')
    df = pd.DataFrame([{'subject': email_subject, 'body_text': email_body}])
    df = add_url_numeric_features(df)
    df['text'] = build_text_column(df)
    X_text = vec.transform(df['text'])
    numeric = df[['num_urls','num_ip_urls','num_shorteners','num_punycode']].fillna(0).values
    X_numeric = sparse.csr_matrix(numeric)
    X = hstack([X_text, X_numeric])
    pred = clf.predict(X)[0]
    prob = clf.predict_proba(X)[0,1] if hasattr(clf, 'predict_proba') else None
    return {'label': 'phish' if pred==1 else 'legit', 'score': float(prob) if prob is not None else None}
