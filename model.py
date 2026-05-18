import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

def load_model():
    data = pd.read_csv("train.csv")

    data = data[data['label'] != 'label']
    data['label'] = data['label'].astype(str)
    data = data[data['label'].isin(['0','1',0,1])]
    data['label'] = data['label'].astype(int)

    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'[^a-z\s]', '', text)
        return text

    data['sms'] = data['sms'].apply(clean_text)

    X = data['sms']
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,3))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_train_vec, y_train)

    # Metrics
    y_pred = model.predict(X_test_vec)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)

    return model, vectorizer, accuracy, precision, recall, f1, cm


def predict_message(msg, model, vectorizer):
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'[^a-z\s]', '', text)
        return text

    msg_clean = clean_text(msg)

    suspicious_words = ["account","bank","otp","verify","details","kyc","atm"]
    detected = [word for word in suspicious_words if word in msg_clean]

    msg_vec = vectorizer.transform([msg_clean])
    prob = model.predict_proba(msg_vec)[0][1]
    pred = model.predict(msg_vec)[0]

    if pred == 1 or detected:
        return {
            "label": "Spam 🚨",
            "confidence": round(prob*100,2),
            "keywords": detected
        }
    else:
        return {
            "label": "Ham ✅",
            "confidence": round((1-prob)*100,2),
            "keywords": []
        }