# Phishing Email Analyzer

Ready-to-run project skeleton for a phishing email classifier.

## Quick start

1. Create a virtual environment (recommended) and activate it:
```bash
python3 -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate  # Windows PowerShell
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Train a baseline model (this will save model and vectorizer under `models/`):
```bash
python src/train_model.py
```

4. Run the web app:
```bash
python app/main.py
# then open http://127.0.0.1:5000 in your browser
```

## Structure

See the project tree in the repo. `data/` contains small sample CSVs to get started.

