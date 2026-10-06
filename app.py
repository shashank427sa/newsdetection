import os
import re
import joblib
import nltk
import streamlit as st

from datetime import datetime

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #9ca3af;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .result-card {
        padding: 35px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 25px;
    }

    .fake-card {
        background-color: #3b1010;
        border: 1px solid #ef4444;
    }

    .real-card {
        background-color: #0d3524;
        border: 1px solid #22c55e;
    }

    .uncertain-card {
        background-color: #3b2b08;
        border: 1px solid #f59e0b;
    }

    .result-title {
        font-size: 32px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# NLTK SETUP
# ============================================================

try:

    stop_words = set(
        stopwords.words("english")
    )

except LookupError:

    nltk.download("stopwords")

    stop_words = set(
        stopwords.words("english")
    )


try:

    lemmatizer = WordNetLemmatizer()

    lemmatizer.lemmatize("test")

except LookupError:

    nltk.download("wordnet")
    nltk.download("omw-1.4")

    lemmatizer = WordNetLemmatizer()


# ============================================================
# MODEL FILE PATHS
# ============================================================

MODEL_PATH = (
    "models/fake_news_model.pkl"
)

VECTORIZER_PATH = (
    "models/tfidf_vectorizer.pkl"
)

METADATA_PATH = (
    "models/model_metadata.pkl"
)


# ============================================================
# CHECK MODEL FILES
# ============================================================

required_files = [
    MODEL_PATH,
    VECTORIZER_PATH,
    METADATA_PATH
]

missing_files = [
    file
    for file in required_files
    if not os.path.exists(file)
]


if missing_files:

    st.error(
        "❌ Required model files are missing."
    )

    st.write(
        "Run the following command first:"
    )

    st.code(
        "python train_model.py"
    )

    st.write(
        "Missing files:"
    )

    for file in missing_files:

        st.write(
            f"- {file}"
        )

    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = joblib.load(
        MODEL_PATH
    )

    vectorizer = joblib.load(
        VECTORIZER_PATH
    )

    metadata = joblib.load(
        METADATA_PATH
    )

except Exception as e:

    st.error(
        "❌ Could not load the trained model."
    )

    st.code(
        str(e)
    )

    st.stop()


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def clean_text(text):

    text = str(text).lower()

    # Remove URLs

    text = re.sub(
        r"http\S+|www\S+|https\S+",
        " ",
        text
    )

    # Remove HTML

    text = re.sub(
        r"<.*?>",
        " ",
        text
    )

    # Remove special characters

    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    # Remove extra spaces

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

            cleaned_words.append(
                lemmatizer.lemmatize(
                    word
                )
            )

    return " ".join(
        cleaned_words
    )


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:

    st.session_state.history = []


if "sample_title" not in st.session_state:

    st.session_state.sample_title = ""


if "sample_article" not in st.session_state:

    st.session_state.sample_article = ""


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "📰 Fake News Detector"
    )

    st.markdown(
        "---"
    )

    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    st.subheader(
        "📊 Model Information"
    )

    st.write(
        f"**Algorithm:** "
        f"{metadata['model']}"
    )

    st.write(
        f"**Vectorizer:** "
        f"{metadata['vectorizer']}"
    )

    st.write(
        f"**Dataset:** "
        f"{metadata['total_samples']} articles"
    )

    st.write(
        f"**Training:** "
        f"{metadata['training_samples']} articles"
    )

    st.write(
        f"**Testing:** "
        f"{metadata['testing_samples']} articles"
    )

    st.write(
        f"**Features:** "
        f"{metadata['features']}"
    )

    st.markdown(
        "---"
    )

    # --------------------------------------------------------
    # MODEL PERFORMANCE
    # --------------------------------------------------------

    st.subheader(
        "📈 Model Performance"
    )

    st.metric(
        "Accuracy",
        f"{metadata['accuracy'] * 100:.2f}%"
    )

    st.metric(
        "Precision",
        f"{metadata['precision'] * 100:.2f}%"
    )

    st.metric(
        "Recall",
        f"{metadata['recall'] * 100:.2f}%"
    )

    st.metric(
        "F1 Score",
        f"{metadata['f1_score'] * 100:.2f}%"
    )

    st.markdown(
        "---"
    )

    # --------------------------------------------------------
    # CLASSIFICATION INFORMATION
    # --------------------------------------------------------

    st.subheader(
        "⚙️ Classification"
    )

    st.write(
        "The system uses three possible results:"
    )

    st.write(
        "🚨 **FAKE NEWS**"
    )

    st.write(
        "✅ **REAL NEWS**"
    )

    st.write(
        "⚠️ **UNCERTAIN**"
    )

    st.markdown(
        "---"
    )

    # --------------------------------------------------------
    # CLEAR HISTORY
    # --------------------------------------------------------

    if st.button(
        "🗑️ Clear Prediction History",
        use_container_width=True
    ):

        st.session_state.history = []

        st.rerun()


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '📰 Fake News Detection System'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning + Natural Language Processing'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# INTRODUCTION
# ============================================================

st.info(
    "Enter a news title and article below. "
    "The machine-learning model will analyze the "
    "text and classify it as Fake News, Real News, "
    "or Uncertain."
)


# ============================================================
# SAMPLE NEWS
# ============================================================

st.subheader(
    "🧪 Try an Example"
)

sample_col1, sample_col2 = st.columns(2)


# ------------------------------------------------------------
# FAKE EXAMPLE
# ------------------------------------------------------------

fake_sample_title = (
    "BREAKING: Scientists Discover a Secret City Under the Moon"
)

fake_sample_article = (
    "A viral internet report claims that scientists "
    "have discovered a huge secret city beneath the "
    "surface of the Moon. The report says officials "
    "are hiding the discovery from the public. "
    "The claim has not been independently verified "
    "and no reliable evidence has been provided."
)


# ------------------------------------------------------------
# REAL EXAMPLE
# ------------------------------------------------------------

real_sample_title = (
    "Researchers Publish Study on Urban Air Quality"
)

real_sample_article = (
    "Researchers have published a study examining "
    "practical strategies cities can use to reduce "
    "air pollution and improve public health. "
    "The study analyzes air quality measurements "
    "and discusses approaches that local authorities "
    "can consider when developing environmental policies."
)


with sample_col1:

    if st.button(
        "🚨 Load Fake Example",
        use_container_width=True
    ):

        st.session_state.sample_title = (
            fake_sample_title
        )

        st.session_state.sample_article = (
            fake_sample_article
        )

        st.rerun()


with sample_col2:

    if st.button(
        "✅ Load Real Example",
        use_container_width=True
    ):

        st.session_state.sample_title = (
            real_sample_title
        )

        st.session_state.sample_article = (
            real_sample_article
        )

        st.rerun()


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown(
    "---"
)

col1, col2 = st.columns(
    [1, 2]
)


with col1:

    title = st.text_input(
        "📝 News Title",
        value=st.session_state.sample_title,
        placeholder="Enter news headline..."
    )


with col2:

    article = st.text_area(
        "📄 News Article",
        value=st.session_state.sample_article,
        height=200,
        placeholder="Paste the complete news article here..."
    )


# ============================================================
# DETECT BUTTON
# ============================================================

st.markdown(
    ""
)

detect_button = st.button(
    "🔍 DETECT NEWS",
    use_container_width=True,
    type="primary"
)


# ============================================================
# PREDICTION
# ============================================================

if detect_button:

    # --------------------------------------------------------
    # VALIDATE INPUT
    # --------------------------------------------------------

    if (
        not title.strip()
        and not article.strip()
    ):

        st.warning(
            "⚠️ Please enter a news title or article."
        )

        st.stop()


    # --------------------------------------------------------
    # COMBINE TITLE + ARTICLE
    # --------------------------------------------------------

    combined_text = (
        title + " " + article
    )


    # --------------------------------------------------------
    # CLEAN TEXT
    # --------------------------------------------------------

    cleaned = clean_text(
        combined_text
    )


    if not cleaned:

        st.warning(
            "⚠️ The text could not be processed."
        )

        st.stop()


    # --------------------------------------------------------
    # TF-IDF TRANSFORMATION
    # --------------------------------------------------------

    vector = vectorizer.transform(
        [cleaned]
    )


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(
        vector
    )[0]


    # --------------------------------------------------------
    # MODEL PROBABILITY
    #
    # Used internally only to decide whether the
    # prediction should be FAKE, REAL, or UNCERTAIN.
    #
    # It is NOT displayed to the user.
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        vector
    )[0]


    confidence = max(
        probabilities
    )


    threshold = (
        metadata["uncertain_threshold"]
    )


    # --------------------------------------------------------
    # FINAL CLASSIFICATION
    # --------------------------------------------------------

    if confidence < threshold:

        result = "UNCERTAIN"

        result_icon = "⚠️"

        result_class = (
            "uncertain-card"
        )

    elif prediction == 0:

        result = "FAKE NEWS"

        result_icon = "🚨"

        result_class = (
            "fake-card"
        )

    else:

        result = "REAL NEWS"

        result_icon = "✅"

        result_class = (
            "real-card"
        )


    # ========================================================
    # SAVE HISTORY
    # ========================================================

    history_item = {

        "time": datetime.now().strftime(
            "%H:%M:%S"
        ),

        "title": (
            title
            if title.strip()
            else "No title"
        ),

        "result": result
    }


    st.session_state.history.insert(
        0,
        history_item
    )


    # Keep only latest 20

    st.session_state.history = (
        st.session_state.history[:20]
    )


    # ========================================================
    # RESULT
    # ========================================================

    st.markdown(
        "---"
    )

    st.markdown(
        f"""
        <div class="result-card {result_class}">
            <div class="result-title">
                {result_icon} {result}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # RESULT EXPLANATION
    # ========================================================

    if result == "UNCERTAIN":

        st.warning(
            "⚠️ The model is not sufficiently confident "
            "to classify this article as definitely "
            "Fake or Real."
        )

    elif result == "FAKE NEWS":

        st.error(
            "🚨 The model classified this article "
            "as Fake News based on patterns learned "
            "from the training dataset."
        )

    else:

        st.success(
            "✅ The model classified this article "
            "as Real News based on patterns learned "
            "from the training dataset."
        )


    st.caption(
        "⚠️ This is an educational machine-learning "
        "classifier and should not be treated as "
        "definitive fact verification."
    )


# ============================================================
# PREDICTION HISTORY
# ============================================================

st.markdown(
    "---"
)

st.subheader(
    "🕘 Prediction History"
)


if len(
    st.session_state.history
) == 0:

    st.info(
        "No predictions yet. "
        "Your predictions will appear here."
    )

else:

    for item in st.session_state.history:

        if item["result"] == "FAKE NEWS":

            icon = "🚨"

        elif item["result"] == "REAL NEWS":

            icon = "✅"

        else:

            icon = "⚠️"


        history_col1, history_col2, history_col3 = (
            st.columns(
                [1, 6, 2]
            )
        )


        with history_col1:

            st.write(
                icon
            )


        with history_col2:

            st.write(
                f"**{item['title']}**"
            )

            st.caption(
                item["time"]
            )


        with history_col3:

            st.write(
                item["result"]
            )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown(
    "---"
)

st.subheader(
    "📈 Model Performance"
)


graph_col1, graph_col2 = st.columns(2)


# ------------------------------------------------------------
# ACCURACY GRAPH
# ------------------------------------------------------------

with graph_col1:

    graph_path = (
        "outputs/accuracy_graph.png"
    )

    if os.path.exists(
        graph_path
    ):

        st.image(
            graph_path,
            caption="Model Performance Metrics",
            use_container_width=True
        )

    else:

        st.warning(
            "Accuracy graph not found."
        )


# ------------------------------------------------------------
# CONFUSION MATRIX
# ------------------------------------------------------------

with graph_col2:

    matrix_path = (
        "outputs/confusion_matrix.png"
    )

    if os.path.exists(
        matrix_path
    ):

        st.image(
            matrix_path,
            caption="Confusion Matrix",
            use_container_width=True
        )

    else:

        st.warning(
            "Confusion matrix not found."
        )


# ============================================================
# ABOUT PROJECT
# ============================================================

st.markdown(
    "---"
)

with st.expander(
    "ℹ️ About This Project"
):

    st.write(
        """
        This project demonstrates a Fake News Detection
        system using Natural Language Processing and
        Machine Learning.

        The workflow includes:

        • Text preprocessing

        • Stopword removal

        • Lemmatization

        • TF-IDF feature extraction

        • Logistic Regression classification

        • Model evaluation

        • Confusion matrix

        • Accuracy, precision, recall and F1 score

        • Streamlit web deployment

        • Prediction history
        """
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown(
    "---"
)

st.caption(
    "⚠️ Educational Project — The built-in dataset is "
    "synthetic and is intended to demonstrate the "
    "machine-learning workflow. This application is "
    "not a definitive fact-checking service."
)