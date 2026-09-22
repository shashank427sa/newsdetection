import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def load_or_create_data(filepath="news.csv"):
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
    else:
        data = {
            "text": [
                "The scientific community confirms the new telescope discovered an exoplanet with water vapor.",
                "Government signs international trade treaty to reduce carbon emissions by thirty percent.",
                "Central Bank announces interest rate hike of 25 basis points following quarterly review.",
                "Aliens have officially landed in Antarctica and taken control of global leaders secretly!",
                "Drinking boiled lemon peel cured all known diseases overnight according to viral claims.",
                "Secret microchips discovered in tap water supply across all major cities worldwide!",
                "Stock market indices hit all-time high following strong corporate tech earnings.",
                "NASA rover discovers ancient riverbed formations on Mars surface.",
                "Celebrity claims drinking pure gold liquid grants immortality in leaked video.",
                "Ministry of Health issues updated seasonal vaccination recommendations.",
            ],
            "label": [
                "REAL",
                "REAL",
                "REAL",
                "FAKE",
                "FAKE",
                "FAKE",
                "REAL",
                "REAL",
                "FAKE",
                "REAL",
            ],
        }
        df = pd.DataFrame(data)
    return df


def main():
    print("Loading dataset...")
    df = load_or_create_data()

    df = df.dropna(subset=["text", "label"])
    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Vectorizing text data...")
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training Passive-Aggressive Classifier...")
    model = PassiveAggressiveClassifier(max_iter=50, random_state=42, C=0.5)
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    print(f"Validation Accuracy: {acc * 100:.2f}%")

    # Serialize files
    joblib.dump(model, "model.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    print("SUCCESS: 'model.pkl' and 'tfidf_vectorizer.pkl' created.")


if __name__ == "__main__":
    main()