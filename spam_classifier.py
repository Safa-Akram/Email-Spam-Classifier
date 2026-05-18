import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# Step 1: Load dataset
# -------------------------------
data = pd.read_csv("train.csv")

print("Columns in dataset:", data.columns)

# Remove wrong rows (if header repeated)
data = data[data['label'] != 'label']

# Ensure correct label format
data['label'] = data['label'].astype(str)
data = data[data['label'].isin(['0', '1', 0, 1])]
data['label'] = data['label'].astype(int)

# -------------------------------
# Step 2: Clean text
# -------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

data['sms'] = data['sms'].apply(clean_text)

# -------------------------------
# Step 3: Prepare data
# -------------------------------
X = data['sms']
y = data['label']

print("\nLabel Distribution:\n", y.value_counts())

# -------------------------------
# Step 4: Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------
# Step 5: TF-IDF Vectorization
# -------------------------------
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1,3),
    max_features=10000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -------------------------------
# Step 6: Train Model
# -------------------------------
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train_vec, y_train)

# -------------------------------
# Step 7: Evaluate Model
# -------------------------------
y_pred = model.predict(X_test_vec)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# Step 8: Rule-based system
# -------------------------------
def rule_based_check(text):
    suspicious_words = [
        "account", "bank", "otp", "verify", "details",
        "credit", "loan", "kyc", "atm", "pin"
    ]

    for word in suspicious_words:
        if word in text:
            return 1
    return 0

# -------------------------------
# Step 9: User Input
# -------------------------------
while True:
    msg = input("\nEnter a message (or type 'exit'): ")

    if msg.lower() == "exit":
        break

    msg_clean = clean_text(msg)

    # Rule-based override
    if rule_based_check(msg_clean) == 1:
        print("Prediction: Spam 🚨")
        continue

    msg_vec = vectorizer.transform([msg_clean])
    prediction = model.predict(msg_vec)[0]

    if prediction == 1:
        print("Prediction: Spam 🚨")
    else:
        print("Prediction: Ham ✅")