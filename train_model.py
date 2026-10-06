import os
import re
import random
import joblib
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs("dataset", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# ============================================================
# NLTK
# ============================================================

print("\nDownloading NLTK resources...")

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

stop_words = set(
    stopwords.words("english")
)

lemmatizer = WordNetLemmatizer()


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"http\S+|www\S+|https\S+",
        " ",
        text
    )

    text = re.sub(
        r"<.*?>",
        " ",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    words = text.split()

    cleaned_words = []

    for word in words:

        if (
            word not in stop_words
            and len(word) > 2
        ):

            word = lemmatizer.lemmatize(
                word
            )

            cleaned_words.append(word)

    return " ".join(
        cleaned_words
    )


# ============================================================
# SYNTHETIC DATASET
# ============================================================
#
# This creates a large educational dataset automatically.
#
# IMPORTANT:
# This is NOT a real-world news dataset.
# ============================================================


topics = [
    "government",
    "technology",
    "health",
    "science",
    "education",
    "business",
    "environment",
    "transport",
    "space",
    "sports"
]


locations = [
    "India",
    "New Delhi",
    "Mumbai",
    "Delhi",
    "Bengaluru",
    "Chennai",
    "Kolkata",
    "Hyderabad",
    "Pune",
    "the country"
]


# ============================================================
# REAL NEWS COMPONENTS
# ============================================================

real_subjects = [
    "government officials",
    "researchers",
    "university researchers",
    "health officials",
    "transport authorities",
    "company representatives",
    "education officials",
    "scientists",
    "local authorities",
    "industry analysts"
]


real_verbs = [
    "announced",
    "published",
    "reported",
    "released",
    "approved",
    "introduced",
    "presented",
    "examined",
    "reviewed",
    "confirmed"
]


real_objects = [
    "a new research report",
    "updated public guidance",
    "a new development program",
    "the results of a study",
    "an annual report",
    "a new policy proposal",
    "updated safety measures",
    "a technology update",
    "a public transport plan",
    "a research project"
]


real_details = [
    "The announcement included information about the project and its expected implementation.",
    "Officials said the measure would be reviewed periodically.",
    "Researchers said additional studies are needed to understand the long-term effects.",
    "The report included data collected over several months.",
    "The organization said the program would be implemented in phases.",
    "The findings were presented during a public meeting.",
    "The proposal will be considered by the relevant authorities.",
    "The study examined available evidence and identified several areas for further research.",
    "Officials encouraged the public to use information from reliable sources.",
    "The company said more details would be provided after the review."
]


real_title_templates = [
    "{subject} {verb} {object}",
    "{location}: {subject} {verb} {object}",
    "New report: {subject} {verb} {object}",
    "{subject} {verb} updated information on {object}",
    "{location} authorities {verb} {object}",
]


# ============================================================
# FAKE NEWS COMPONENTS
# ============================================================

fake_subjects = [
    "a viral post",
    "an anonymous source",
    "an internet report",
    "a sensational website",
    "a social media message",
    "an unverified article",
    "a mysterious source",
    "an online claim",
    "a viral video",
    "an unverified report"
]


fake_verbs = [
    "claims",
    "alleges",
    "reportedly reveals",
    "secretly announces",
    "supposedly proves",
    "is said to reveal",
    "mysteriously claims",
    "apparently confirms"
]


fake_objects = [
    "a secret government plan",
    "a miraculous discovery",
    "a hidden technology",
    "a completely unknown city",
    "a revolutionary invention",
    "a secret cure",
    "a shocking scientific discovery",
    "a mysterious experiment",
    "a hidden international agreement",
    "an unbelievable breakthrough"
]


fake_details = [
    "The report provides no verifiable evidence for the claim.",
    "The story says officials are hiding the information from the public.",
    "The article asks readers to share the information immediately.",
    "No independent organization has confirmed the claim.",
    "The source does not provide reliable documents or research.",
    "The story contains sensational statements without supporting evidence.",
    "The information has circulated widely on social media.",
    "The report uses anonymous sources without identifying them.",
    "The claim has not been independently verified.",
    "The article encourages readers to believe the information without providing reliable evidence."
]


fake_title_templates = [
    "BREAKING: {subject} {verb} {object}",
    "SHOCKING: {object} {verb} by {subject}",
    "{subject} {verb} {object}",
    "URGENT: {object} reportedly discovered",
    "VIRAL: {subject} {verb} {object}",
    "Secret discovery: {object}",
]


# ============================================================
# GENERATE REAL ARTICLES
# ============================================================

def generate_real_article():

    subject = random.choice(
        real_subjects
    )

    verb = random.choice(
        real_verbs
    )

    obj = random.choice(
        real_objects
    )

    location = random.choice(
        locations
    )

    title_template = random.choice(
        real_title_templates
    )

    title = title_template.format(
        subject=subject,
        verb=verb,
        object=obj,
        location=location
    )

    detail_1 = random.choice(
        real_details
    )

    detail_2 = random.choice(
        real_details
    )

    detail_3 = random.choice(
        real_details
    )

    text = (
        f"{subject.capitalize()} {verb} "
        f"{obj}. "
        f"{detail_1} "
        f"{detail_2} "
        f"{detail_3}"
    )

    return {
        "title": title,
        "text": text,
        "subject": random.choice(topics),
        "date": f"2025-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
        "label": 1
    }


# ============================================================
# GENERATE FAKE ARTICLES
# ============================================================

def generate_fake_article():

    subject = random.choice(
        fake_subjects
    )

    verb = random.choice(
        fake_verbs
    )

    obj = random.choice(
        fake_objects
    )

    title_template = random.choice(
        fake_title_templates
    )

    title = title_template.format(
        subject=subject,
        verb=verb,
        object=obj
    )

    detail_1 = random.choice(
        fake_details
    )

    detail_2 = random.choice(
        fake_details
    )

    detail_3 = random.choice(
        fake_details
    )

    text = (
        f"{subject.capitalize()} "
        f"{verb} {obj}. "
        f"{detail_1} "
        f"{detail_2} "
        f"{detail_3}"
    )

    return {
        "title": title,
        "text": text,
        "subject": random.choice(topics),
        "date": f"2025-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
        "label": 0
    }


# ============================================================
# BUILD DATASET
# ============================================================

print("\nGenerating dataset...")

dataset = []


# 750 fake articles
for _ in range(750):

    dataset.append(
        generate_fake_article()
    )


# 750 real articles
for _ in range(750):

    dataset.append(
        generate_real_article()
    )


random.shuffle(
    dataset
)


df = pd.DataFrame(
    dataset
)


print(
    "\nDataset created successfully."
)

print(
    "Total articles:",
    len(df)
)

print(
    "Fake articles:",
    len(
        df[df["label"] == 0]
    )
)

print(
    "Real articles:",
    len(
        df[df["label"] == 1]
    )
)


# ============================================================
# SAVE DATASET
# ============================================================

df.to_csv(
    "dataset/news.csv",
    index=False
)


# ============================================================
# COMBINE TITLE + ARTICLE
# ============================================================

df["content"] = (
    df["title"].fillna("") +
    " " +
    df["text"].fillna("")
)


# ============================================================
# CLEAN TEXT
# ============================================================

print(
    "\nCleaning text..."
)

df["clean_content"] = (
    df["content"].apply(
        clean_text
    )
)


# ============================================================
# FEATURES AND LABEL
# ============================================================

X = df["clean_content"]

y = df["label"]


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_SEED,
    stratify=y
)


print(
    "\nTraining samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# ============================================================
# TF-IDF
# ============================================================

print(
    "\nCreating TF-IDF vectors..."
)


vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=20000,
    min_df=2,
    sublinear_tf=True
)


X_train_tfidf = (
    vectorizer.fit_transform(
        X_train
    )
)


X_test_tfidf = (
    vectorizer.transform(
        X_test
    )
)


print(
    "Number of TF-IDF features:",
    X_train_tfidf.shape[1]
)


# ============================================================
# TRAIN MODEL
# ============================================================

print(
    "\nTraining Logistic Regression..."
)


model = LogisticRegression(
    max_iter=3000,
    random_state=RANDOM_SEED,
    class_weight="balanced"
)


model.fit(
    X_train_tfidf,
    y_train
)


# ============================================================
# PREDICTIONS
# ============================================================

predictions = model.predict(
    X_test_tfidf
)


probabilities = model.predict_proba(
    X_test_tfidf
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)


print(
    "\n=========================================="
)

print(
    "MODEL PERFORMANCE"
)

print(
    "=========================================="
)

print(
    f"Accuracy  : {accuracy * 100:.2f}%"
)

print(
    f"Precision : {precision * 100:.2f}%"
)

print(
    f"Recall    : {recall * 100:.2f}%"
)

print(
    f"F1 Score  : {f1 * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "FAKE",
            "REAL"
        ],
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    predictions
)


plt.figure(
    figsize=(7, 6)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "Fake News Detection - Confusion Matrix"
)

plt.colorbar()

plt.xticks(
    [0, 1],
    ["FAKE", "REAL"]
)

plt.yticks(
    [0, 1],
    ["FAKE", "REAL"]
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "Actual Label"
)


for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrix.png",
    dpi=150
)

plt.close()


# ============================================================
# ACCURACY GRAPH
# ============================================================

metrics = [
    accuracy,
    precision,
    recall,
    f1
]

metric_names = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]


plt.figure(
    figsize=(8, 5)
)

bars = plt.bar(
    metric_names,
    metrics
)


plt.ylim(
    0,
    1
)

plt.title(
    "Model Performance"
)

plt.ylabel(
    "Score"
)


for bar, value in zip(
    bars,
    metrics
):

    plt.text(
        bar.get_x()
        + bar.get_width() / 2,
        value + 0.02,
        f"{value * 100:.1f}%",
        ha="center"
    )


plt.tight_layout()

plt.savefig(
    "outputs/accuracy_graph.png",
    dpi=150
)

plt.close()


# ============================================================
# SAVE RESULTS
# ============================================================

results = pd.DataFrame({

    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],

    "Score": [
        accuracy,
        precision,
        recall,
        f1
    ]
})


results.to_csv(
    "outputs/model_results.csv",
    index=False
)


# ============================================================
# SAVE MODEL
# ============================================================

print(
    "\nSaving model..."
)


joblib.dump(
    model,
    "models/fake_news_model.pkl"
)


joblib.dump(
    vectorizer,
    "models/tfidf_vectorizer.pkl"
)


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {

    "accuracy": accuracy,

    "precision": precision,

    "recall": recall,

    "f1_score": f1,

    "training_samples": len(X_train),

    "testing_samples": len(X_test),

    "total_samples": len(df),

    "features": X_train_tfidf.shape[1],

    "model": "Logistic Regression",

    "vectorizer": "TF-IDF",

    "uncertain_threshold": 0.65
}


joblib.dump(
    metadata,
    "models/model_metadata.pkl"
)


# ============================================================
# FINISHED
# ============================================================

print(
    "\n=========================================="
)

print(
    "TRAINING COMPLETED SUCCESSFULLY"
)

print(
    "=========================================="
)

print(
    "\nCreated:"
)

print(
    "dataset/news.csv"
)

print(
    "models/fake_news_model.pkl"
)

print(
    "models/tfidf_vectorizer.pkl"
)

print(
    "models/model_metadata.pkl"
)

print(
    "outputs/confusion_matrix.png"
)

print(
    "outputs/accuracy_graph.png"
)

print(
    "outputs/model_results.csv"
)

print(
    "\nRun the application using:"
)

print(
    "python -m streamlit run app.py"
)