# AI Spam & Phishing Detector (College Project)

Detects: **spam emails**, **phishing emails**, **phishing links**.

## Run
    pip install -r requirements.txt
    python detector.py

## How it works
| Part | Technique |
|------|-----------|
| Email classifier (legitimate / spam / phishing) | TF-IDF text features + Naive Bayes |
| Link detector | Hand-made URL features (length, IP address, '@', keywords, shortener, risky TLD...) + Random Forest, blended with simple rules |
| Combined check | Links found inside an email are also scanned; a phishing link upgrades the email verdict |

## Make it more accurate (important for marks)
The built-in dataset is tiny, only for demo. Replace it with real data:
- Emails: SpamAssassin Public Corpus, Enron Spam, Kaggle "Phishing Email Dataset", SMS Spam Collection
- URLs: PhishTank, OpenPhish (phishing) + Tranco top sites list (safe)

Load a CSV with pandas and pass `(text, label)` pairs to `train_email_model()`,
and `(url, 0/1)` pairs to `train_url_model()`. Then report accuracy,
precision, recall with `train_test_split` + `classification_report`.

## Ideas to extend
Streamlit/Flask web UI, Gmail inbox scanning, WHOIS domain age, Google Safe Browsing API.

## Limitations (mention in your report)
Small dataset, no live reputation lookup, can be fooled by new attack styles.
