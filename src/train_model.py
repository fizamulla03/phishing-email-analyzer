import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from .feature_extraction import build_text_column, fit_vectorizer, add_url_numeric_features
import os
from scipy import sparse
from scipy.sparse import hstack

def load_data():
    # simple merge of sample CSVs (user can replace with larger datasets)
    p = pd.read_csv('data/phishing_emails.csv')
    l = pd.read_csv('data/legit_emails.csv')
    df = pd.concat([p, l], ignore_index=True)
    return df

def main():
    df = load_data()
    df = add_url_numeric_features(df)
    df['text'] = build_text_column(df)

    # vectorize text
    vec = fit_vectorizer(df['text'], max_features=2000)
    X_text = vec.transform(df['text'])

    # combine text features and numeric URL features
    numeric = df[['num_urls','num_ip_urls','num_shorteners','num_punycode']].fillna(0).values
    X_numeric = sparse.csr_matrix(numeric)
    X = hstack([X_text, X_numeric])
    y = df['label'].map({'phish':1,'legit':0}).values

    # If dataset is tiny, skip stratified train/test split to avoid errors
    try:
        if len(y) < 10:
            print("Dataset small (less than 10 rows) — training on all data without test split for demo.")
            X_train, X_test, y_train, y_test = X, X, y, y
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    except ValueError:
        # fallback: do a non-stratified split
        print("Stratified split failed — falling back to non-stratified split.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    clf = RandomForestClassifier(n_estimators=200, class_weight='balanced', n_jobs=-1, random_state=42)
    print('Training model...')
    clf.fit(X_train, y_train)

    # Evaluate only if we have a meaningful test set
    if X_test is not X_train:
        pred = clf.predict(X_test)
        print('\nClassification report:')
        print(classification_report(y_test, pred))
    else:
        print("\nNo separate test set was used (demo mode).")

    os.makedirs('models', exist_ok=True)
    joblib.dump(clf, 'models/model.pkl')
    joblib.dump(vec, 'models/vectorizer.pkl')
    print('Saved model to models/model.pkl and vectorizer to models/vectorizer.pkl')

if __name__ == '__main__':
    main()
