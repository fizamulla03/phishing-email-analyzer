import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from .feature_extraction import build_text_column, fit_vectorizer, add_url_numeric_features
import os

def load_data():
    # simple merge of sample CSVs (user can replace with larger datasets)
    p = pd.read_csv('data/phishing_emails.csv')
    l = pd.read_csv('data/legit_emails.csv')
    df = pd.concat([p,l], ignore_index=True)
    return df

def main():
    df = load_data()
    df = add_url_numeric_features(df)
    df['text'] = build_text_column(df)
    # vectorize text
    vec = fit_vectorizer(df['text'], max_features=2000)
    X_text = vec.transform(df['text'])
    # combine text features and numeric URL features
    from scipy.sparse import hstack
    numeric = df[['num_urls','num_ip_urls','num_shorteners','num_punycode']].fillna(0).values
    from scipy import sparse
    X_numeric = sparse.csr_matrix(numeric)
    X = hstack([X_text, X_numeric])
    y = df['label'].map({'phish':1,'legit':0}).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, class_weight='balanced', n_jobs=-1, random_state=42)
    print('Training model...')
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print('\nClassification report:')
    print(classification_report(y_test, pred))

    os.makedirs('models', exist_ok=True)
    joblib.dump(clf, 'models/model.pkl')
    joblib.dump(vec, 'models/vectorizer.pkl')
    print('Saved model to models/model.pkl and vectorizer to models/vectorizer.pkl')

if __name__ == '__main__':
    main()
