"""
Simple AI Spam Email, Phishing Email & Phishing Link Detector
College project - uses scikit-learn.

Run:  python detector.py
"""
import re
from urllib.parse import urlparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# ----------------------------------------------------------------------
# 1. EMAIL DATA  (tiny demo set - replace with a real dataset, see README)
# ----------------------------------------------------------------------
EMAILS = [
    # legitimate
    ("Hi team, the meeting is moved to 3pm tomorrow. Please bring your project reports.", "legitimate"),
    ("Your assignment has been graded. Check the course portal for feedback.", "legitimate"),
    ("Lunch on Friday? Let me know if you are free, we can try the new cafe.", "legitimate"),
    ("Attached is the agenda for next week's project review. Let me know your comments.", "legitimate"),
    ("Thanks for your help with the presentation, it went really well.", "legitimate"),
    ("Reminder: the library will close early on Saturday for maintenance.", "legitimate"),
    ("Can you send me the notes from yesterday's lecture? I missed the class.", "legitimate"),
    ("Your order #4821 has shipped and will arrive on Thursday.", "legitimate"),
    ("Happy birthday! Hope you have a great day with family and friends.", "legitimate"),
    ("Please find the invoice for last month's internship stipend attached.", "legitimate"),
    # spam
    ("WIN a FREE iPhone now!!! Click here to claim your prize, limited offer!!!", "spam"),
    ("Buy cheap pills online, 90% discount, no prescription needed, order today", "spam"),
    ("Congratulations! You have won $1,000,000 lottery. Send details to claim", "spam"),
    ("Earn money from home $5000 per week, no experience, work only 2 hours", "spam"),
    ("Hot singles in your area want to meet you. Sign up free today", "spam"),
    ("Lowest price guaranteed! Buy now, huge sale, free shipping, act fast", "spam"),
    ("Make easy cash fast, double your investment in 24 hours guaranteed", "spam"),
    ("Get a free gift card, you are our lucky winner, click to claim reward", "spam"),
    ("Lose 10 kg in 7 days with this miracle weight loss pill, order now", "spam"),
    ("Exclusive deal just for you: casino bonus, free spins, deposit now", "spam"),
    # phishing
    ("Urgent: your bank account has been suspended. Verify your password immediately to restore access.", "phishing"),
    ("Security alert: unusual login detected. Confirm your identity by entering your credentials here.", "phishing"),
    ("Your PayPal account is limited. Update your payment information within 24 hours or it will be closed.", "phishing"),
    ("Dear customer, your card was blocked. Click the link to verify your account number and PIN.", "phishing"),
    ("Your Microsoft 365 password expires today. Sign in now to keep your mailbox active.", "phishing"),
    ("IT Helpdesk: we are upgrading mail servers. Reply with your username and password to avoid deactivation.", "phishing"),
    ("Your Netflix payment failed. Update billing details now to avoid account termination.", "phishing"),
    ("Tax refund pending. Confirm your bank details and social security number to receive payment.", "phishing"),
    ("Final notice: unauthorized transaction on your account. Login to verify or your account will be locked.", "phishing"),
    ("Your package could not be delivered. Pay the small customs fee and confirm your address and card.", "phishing"),
]

# ----------------------------------------------------------------------
# 2. URL DATA  (0 = safe, 1 = phishing)
# ----------------------------------------------------------------------
URLS = [
    ("https://www.google.com", 0), ("https://github.com/torvalds/linux", 0),
    ("https://en.wikipedia.org/wiki/Machine_learning", 0), ("https://www.amazon.in/dp/B08N5WRWNW", 0),
    ("https://stackoverflow.com/questions/12345/python-list", 0), ("https://www.youtube.com/watch?v=abc123", 0),
    ("https://mail.google.com/mail/u/0/", 0), ("https://www.linkedin.com/in/someone", 0),
    ("https://docs.python.org/3/tutorial/", 0), ("https://www.flipkart.com/mobiles", 0),
    ("https://www.paypal.com/signin", 0), ("https://www.microsoft.com/en-in/windows", 0),
    ("http://paypal-secure-login.verify-account.xyz/signin?id=88213", 1),
    ("http://192.168.45.12/bank/login.php", 1),
    ("http://amazon.account-update.support-center.top/verify", 1),
    ("https://secure-sbi-netbanking-login.com.co/auth@confirm", 1),
    ("http://bit.ly/3xYz9Qa", 1),
    ("http://microsoft365-password-expire.info/login/update-now", 1),
    ("http://faceb00k-security-check.tk/confirm-account", 1),
    ("http://netflix-billing-update.verify-payment.ml/signin", 1),
    ("http://45.77.12.9:8080/webscr/cmd=login-submit", 1),
    ("https://apple-id-locked.account-verify-secure.xyz/unlock", 1),
    ("http://hdfcbank-kyc-update.online/login/verify-otp", 1),
    ("http://goo-gle.com-signin.verify.work/password", 1),
]

SUSPICIOUS_WORDS = ["login", "signin", "verify", "update", "secure", "account",
                    "password", "confirm", "bank", "kyc", "otp", "webscr", "unlock"]
SHORTENERS = {"bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "cutt.ly"}
BAD_TLDS = {"xyz", "top", "tk", "ml", "ga", "cf", "gq", "work", "online", "support", "info"}


# ----------------------------------------------------------------------
# 3. URL FEATURE EXTRACTION
# ----------------------------------------------------------------------
def url_features(url):
    parsed = urlparse(url if "://" in url else "http://" + url)
    host = parsed.netloc.lower().split(":")[0]
    low = url.lower()
    tld = host.split(".")[-1] if "." in host else ""
    return [
        len(url),                                              # long URLs are suspicious
        host.count("."),                                       # many subdomains
        url.count("-"),                                        # hyphens
        int("@" in url),                                       # '@' trick
        int(bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host))),  # IP instead of domain
        int(parsed.scheme == "https"),
        sum(c.isdigit() for c in url) / max(len(url), 1),      # digit ratio
        sum(w in low for w in SUSPICIOUS_WORDS),               # keywords
        int(host in SHORTENERS),
        int(tld in BAD_TLDS),
        int(bool(parsed.port)),                                # odd port
    ]


URL_REASONS = ["very long URL", "many subdomains", "many hyphens", "contains '@'",
               "uses IP address", "uses HTTPS", "many digits", "suspicious keywords",
               "URL shortener", "risky domain ending", "unusual port"]


def explain_url(url):
    f = url_features(url)
    out = []
    if f[0] > 60: out.append("very long URL")
    if f[1] >= 3: out.append("many subdomains")
    if f[2] >= 2: out.append("many hyphens")
    if f[3]: out.append("contains '@'")
    if f[4]: out.append("uses IP address instead of a domain")
    if not f[5]: out.append("not using HTTPS")
    if f[7] >= 2: out.append("suspicious keywords (login/verify/update...)")
    if f[8]: out.append("URL shortener hides the real destination")
    if f[9]: out.append("risky domain ending")
    if f[10]: out.append("unusual port")
    return out


# ----------------------------------------------------------------------
# 4. TRAIN MODELS
# ----------------------------------------------------------------------
def train_email_model(data=EMAILS):
    texts, labels = zip(*data)
    model = make_pipeline(TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2)),
                          MultinomialNB(alpha=0.5))
    model.fit(texts, labels)
    return model


def train_url_model(data=URLS):
    X = [url_features(u) for u, _ in data]
    y = [l for _, l in data]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model


# ----------------------------------------------------------------------
# 5. PREDICTION HELPERS
# ----------------------------------------------------------------------
def find_urls(text):
    return re.findall(r"(?:https?://|www\.)[^\s<>\"']+", text)


def check_url(model, url):
    prob = model.predict_proba([url_features(url)])[0][1]
    # blend ML with simple rules so tiny training data still behaves sensibly
    reasons = explain_url(url)
    score = 0.7 * prob + 0.3 * min(len(reasons) / 4, 1)
    verdict = "PHISHING" if score >= 0.5 else "SAFE"
    return verdict, score, reasons


def check_email(email_model, url_model, text):
    probs = email_model.predict_proba([text])[0]
    classes = email_model.classes_
    label = classes[probs.argmax()]
    result = {"label": label, "confidence": probs.max(), "links": []}
    for u in find_urls(text):
        result["links"].append((u, *check_url(url_model, u)))
    # any phishing link in an email upgrades the verdict
    if any(v == "PHISHING" for _, v, _, _ in result["links"]) and label == "legitimate":
        result["label"] = "phishing"
        result["note"] = "Email text looked normal, but it contains a phishing link."
    return result


# ----------------------------------------------------------------------
# 6. COMMAND LINE INTERFACE
# ----------------------------------------------------------------------
def main():
    print("Training models...")
    email_model, url_model = train_email_model(), train_url_model()
    print("Ready!\n")
    while True:
        print("=" * 50)
        print(" 1) Check an email   2) Check a link   3) Quit")
        choice = input("Choose: ").strip()
        if choice == "1":
            print("Paste the email text (one line):")
            r = check_email(email_model, url_model, input("> "))
            print(f"\nResult : {r['label'].upper()}  (confidence {r['confidence']:.0%})")
            if "note" in r: print("Note   :", r["note"])
            for u, v, s, why in r["links"]:
                print(f"Link   : {u} -> {v} ({s:.0%}) {', '.join(why)}")
        elif choice == "2":
            u = input("URL: ").strip()
            v, s, why = check_url(url_model, u)
            print(f"\nResult : {v}  (risk {s:.0%})")
            if why: print("Why    :", "; ".join(why))
        elif choice == "3":
            break
        print()


if __name__ == "__main__":
    main()
