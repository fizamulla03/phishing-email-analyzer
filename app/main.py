from flask import Flask, render_template, request
import joblib
import os
import pandas as pd
from src.feature_extraction import build_text_column, add_url_numeric_features
from src.parse_email import extract_urls_from_text, extract_urls_from_html
from scipy.sparse import hstack
from scipy import sparse

# ------------------------------
# 1️⃣ Initialize Flask app
# ------------------------------
app = Flask(__name__)

# ------------------------------
# 2️⃣ Model paths
# ------------------------------
MODEL_PATH = 'models/model.pkl'
VEC_PATH = 'models/vectorizer.pkl'

model = None
vec = None

# ------------------------------
# 3️⃣ Function to load models
# ------------------------------
def load_models():
    global model, vec
    if model is None:
        if not (os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH)):
            raise FileNotFoundError(
                "Model not found. Run `python -m src.train_model` first."
            )
        model = joblib.load(MODEL_PATH)
        vec = joblib.load(VEC_PATH)

# ------------------------------
# 4️⃣ Trusted domains list
# ------------------------------
trusted_domains = [
    "amazon.com", "aws.amazon.com", "google.com", "microsoft.com",
    "linkedin.com", "coursera.org", "udemy.com", "ibm.com",
    "netflix.com", "spotify.com", "slack.com", "github.com",
    "tenable.com", "nessus.com"
]

# ------------------------------
# 5️⃣ Flask route
# ------------------------------
@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    details = {}

    try:
        load_models()
    except Exception as e:
        return f"<h3>Error: {e}</h3><p>Run training first: <code>python -m src.train_model</code></p>"

    if request.method == 'POST':
        email_text = request.form.get('email_text', '')

        # Split subject and body
        lines = email_text.splitlines()
        subject = lines[0] if lines else ''
        body = '\n'.join(lines[1:]) if len(lines) > 1 else email_text

        # Convert to DataFrame
        df = pd.DataFrame([{'subject': subject, 'body_text': body}])

        # Extract features
        df = add_url_numeric_features(df)
        df['text'] = build_text_column(df)

        # Transform features
        X_text = vec.transform(df['text'])
        numeric = df[['num_urls', 'num_ip_urls', 'num_shorteners', 'num_punycode']].fillna(0).values
        X_numeric = sparse.csr_matrix(numeric)
        X = hstack([X_text, X_numeric])

        # Predict
        pred = model.predict(X)[0]

        # --- Trusted domain check ---
        trusted_found = any(domain in body.lower() for domain in trusted_domains)
        if pred == 1 and trusted_found:
            pred = 0

        # Probability if available
        prob = model.predict_proba(X)[0, 1] if hasattr(model, 'predict_proba') else None

        # Final output
        result = '🚨 Phishing Email' if pred == 1 else '✅ Legitimate Email'
        details = {
            'score': float(prob) if prob is not None else None,
            'num_urls': int(df['num_urls'].iloc[0]),
        }
        if trusted_found:
            details['reason'] = 'Found trusted domain in email content.'

    return render_template('index.html', result=result, details=details)

# ------------------------------
# 6️⃣ Run Flask app
# ------------------------------
if __name__ == '__main__':
    app.run(debug=True)
