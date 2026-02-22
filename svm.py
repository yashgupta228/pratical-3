import streamlit as st
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("Spam Email Detector")

emails = [
    "Win a free iPhone now",
    "Meeting at 11 am tomorrow",
    "Congratulations you won lottery",
    "Project discussion with team",
    "Claim your prize immediately",
    "Please find the attached report",
    "Limited offer buy now",
    "Urgent offer expires today",
    "Schedule the meeting for Monday",
    "You have won a cash prize",
    "Monthly performance report attached",
    "Exclusive deal just for you"
]

labels = [1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1]

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(emails)

X_train, X_test, y_train, y_test = train_test_split(
    X, labels,
    test_size=0.25,
    random_state=42,
    stratify=labels
)

model = LinearSVC()
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
st.write(f"Model Accuracy: {accuracy:.2f}")

user_msg = st.text_area("Enter Email Message")

if st.button("Check"):
    if user_msg.strip() == "":
        st.warning("Please enter a message")
    else:
        msg_vec = vectorizer.transform([user_msg])
        pred = model.predict(msg_vec)[0]

        if pred == 1:
            st.error("Spam Email")
        else:
            st.success("Not Spam Email")
