import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Spam Email Detector", page_icon="📧", layout="centered")

st.title("Spam Email Detector")
st.write("Detect whether an email is Spam or Not Spam using Machine Learning")

# Sample dataset
emails = [
    "Win a free iPhone now",
    "Meeting at 11 am tomorrow",
    "Congratulations you won lottery",
    "Project discussion today",
    "Claim your free prize now",
    "Let's have lunch tomorrow",
    "Exclusive offer just for you",
    "Team meeting agenda attached",
    "Urgent! Update your account now",
    "Family dinner tonight",
    "Limited time discount offer",
    "See you at the conference"
]

labels = [1,0,1,0,1,0,1,0,1,0,1,0]  # 1 = Spam, 0 = Not Spam

@st.cache_resource
def train():
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1,2),
        max_df=0.9,
        min_df=1
    )

    X = vectorizer.fit_transform(emails)

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels,
        test_size=0.25,
        random_state=42,
        stratify=labels
    )

    model = LinearSVC(C=1.0, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))

    return vectorizer, model, acc

vectorizer, model, accuracy = train()

st.info(f"Model Accuracy: {accuracy * 100:.2f}%")

msg = st.text_area("Enter Email Message", height=120)

if st.button("Check"):
    if msg.strip() == "":
        st.warning("Enter a message")
    else:
        vec = vectorizer.transform([msg])
        pred = model.predict(vec)[0]

        if pred == 1:
            st.error("Result: Spam Email")
        else:
            st.success("Result: Not Spam Email")

st.caption("TF-IDF + Linear SVM Spam Detector")
