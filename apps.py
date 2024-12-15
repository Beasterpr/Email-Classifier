import streamlit as st
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
import re


tdf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))


ps = PorterStemmer()

nltk.download("stopwords")
nltk.download('punkt_tab')

def cleaning(context):
    cleaned_text = re.sub("[^a-zA-Z]", " ", context)  
    cleaned_text = cleaned_text.lower()  
    tokens = nltk.word_tokenize(cleaned_text)  
    stemmed_words = [ps.stem(word) for word in tokens if word not in stopwords.words("english")]
    return " ".join(stemmed_words)


st.set_page_config(
    page_title="Email/SMS Spam Classifier",
    page_icon="📧",
    layout="wide"
)


st.sidebar.title("🛠 Features")
st.sidebar.info("""
- **Input Message:** Type or paste the message to classify as Spam or Not Spam.
- **Spam Analysis:** The app will classify the message as spam or not.
- **Downloadable Report:** You can download the classification results.
""")

st.sidebar.title("📖 About")
st.sidebar.write("""
This app uses machine learning to classify email/SMS messages as **Spam** or **Not Spam**.  
It is powered by a pre-trained model and text preprocessing techniques.
""")

st.title("📧 Email/SMS Spam Classifier 🚀")
st.markdown("""
### Welcome to the Spam Classifier App!  
Paste your email or SMS message below, and let's see if it's spam or not.
""")


st.write("---")
st.subheader("📝 Enter Your Message:")
value = st.text_area("Type your email or SMS here...", height=200)


if st.button("🔍 Predict"):
    if not value.strip():
        st.warning("⚠️ Please enter a message to analyze.")
    else:
       
        with st.spinner("Classifying the message..."):
            transformed_sms = cleaning(value)
            vector_input = tdf.transform([transformed_sms])
            result = model.predict(vector_input)[0]

        if result == 1:
            st.error("💌 **Spam Detected!** This message is classified as spam.")
        else:
            st.success("📩 **Not Spam!** This message is safe.")

        
        confidence = model.predict_proba(vector_input)[0]
        spam_score = round(confidence[1] * 100, 2)
        not_spam_score = round(confidence[0] * 100, 2)

        st.write("---")
        st.subheader("📊 Classification Confidence:")
        st.metric("Spam Probability", f"{spam_score}%", delta=None)
        st.metric("Not Spam Probability", f"{not_spam_score}%", delta=None)

      
        report = f"""
        Email/SMS Spam Classifier Report

        Message: {value.strip()}

        Classification: {"Spam" if result == 1 else "Not Spam"}
        Spam Probability: {spam_score}%
        Not Spam Probability: {not_spam_score}%
        """
        st.write("---")
        st.subheader("📥 Download Classification Report")
        st.download_button(
            label="📄 Download Report as .txt",
            data=report,
            file_name="spam_classifier_report.txt",
            mime="text/plain"
        )


st.write("---")
st.markdown("""
### 🔎 How It Works
1. **Preprocessing**: Cleans and tokenizes the input text.
2. **TF-IDF Vectorization**: Converts text into a numerical format.
3. **Classification**: Uses a pre-trained model to predict if the message is spam.
""")

st.markdown("Made with ❤️ by [Ammar](https://www.instagram.com/sk_.ammar?igsh=ZmJyczE3dzlsc2d6) using [Streamlit](https://streamlit.io/).")

