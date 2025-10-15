from flask import Flask, render_template, request
import joblib
import os
from src.feature_extraction import build_text_column, add_url_numeric_features
from src.parse_email import extract_urls_from_text, extract_urls_from_html
from scipy.sparse import hstack
from scipy import sparse

app = Flask(__name__)
MODEL_PATH = 'models/model.pkl'
VEC_PATH = 'models/vectorizer.pkl'

model = None
vec = None

def load_models():
    global model, vec
    if model is None:
        if not (os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH)):
            raise FileNotFoundError('Model not found. Please run `python src/train_model.py` first to train and save the model.')
        model = joblib.load(MODEL_PATH)
        vec = joblib.load(VEC_PATH)

@app.route('/', methods=['GET','POST'])
def home():
    result = None
    details = {}
    try:
        load_models()
    except Exception as e:
        return f"<h3>Error: {e}</h3><p>Run training first: <code>python src/train_model.py</code></p>"

    if request.method == 'POST':
        email_text = request.form.get('email_text','')
        # naive split: first line as subject if provided
        lines = email_text.splitlines()
        subject = lines[0] if lines else ''
        body = '\n'.join(lines[1:]) if len(lines)>1 else email_text
        df = add_url_numeric_features({'subject':subject,'body_text':body} if isinstance({'subject':subject}, dict) else {})
        # For simplicity, reuse the helper by creating a small DataFrame
        import pandas as pd
        df = pd.DataFrame([{'subject':subject,'body_text':body}])
        df = add_url_numeric_features(df)
        df['text'] = build_text_column(df)
        X_text = vec.transform(df['text'])
        numeric = df[['num_urls','num_ip_urls','num_shorteners','num_punycode']].fillna(0).values
        X_numeric = sparse.csr_matrix(numeric)
        X = hstack([X_text, X_numeric])
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0,1] if hasattr(model, 'predict_proba') else None
        result = 'Phishing Email 🚨' if pred==1 else 'Legitimate Email ✅'
        details = {'score': float(prob) if prob is not None else None, 'num_urls': int(df['num_urls'].iloc[0])}
    return render_template('index.html', result=result, details=details)

if __name__ == '__main__':
    app.run(debug=True)
