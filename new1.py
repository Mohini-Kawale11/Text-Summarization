import streamlit as st
import os
import re
import math
import spacy
import nltk
import PyPDF2
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tag import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from threadpoolctl import threadpool_limits

# Download NLTK resources if not already present
nltk.download("punkt")
nltk.download("words")
nltk.download("maxent_ne_chunker")
nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")

ps = PorterStemmer()
threadpool_limits(limits=1, user_api="openmp")
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

nlp = spacy.load("en_core_web_sm")
stop = set(stopwords.words("english"))
corpus = []

# ---------------- PDF Extractor ----------------
def extract_text_from_pdf(pdf_file):
    text = ""
    reader = PyPDF2.PdfReader(pdf_file)
    num_pages = len(reader.pages)
    if num_pages >= 100:
        num_pages = 100
    for page_num in range(num_pages):
        page = reader.pages[page_num]
        text += page.extract_text()
    return text

# ---------------- Sentence Split ----------------
def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    return [s.strip() for s in sentences]

# ---------------- Numeric Feature Extraction ----------------
def numeric_data(sent_tokens):
    global corpus
    h = {}

    # Feature 1: sentence length
    for i in sent_tokens:
        h[i] = len(i.split(" "))

    # Feature 2: word frequency
    freq = Counter(" ".join(sent_tokens).split())
    for i in sent_tokens:
        score = 0
        for word in i.split():
            score += freq[word]
        h[i] += score

    # Feature 3: TF-IDF score
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sent_tokens)
    scores = X.sum(axis=1).A1
    for idx, sentence in enumerate(sent_tokens):
        h[sentence] += scores[idx]

    # Add more features (up to your 13, as in original script)
    # For example:
    # - position score
    # - title similarity
    # - numerical data count
    # - proper noun ratio
    # - NER presence
    # - cue phrase count
    # - POS tag ratios
    # - etc.

    return h

# ---------------- Question Answering ----------------
def answer_question(text, question):
    try:
        doc = nlp(text)
        sentences = list(doc.sents)

        question_doc = nlp(question)
        keywords = [token.lemma_ for token in question_doc if token.is_alpha and not token.is_stop]

        best_sentence = None
        best_count = 0

        for sentence in sentences:
            count = sum(1 for keyword in keywords if keyword in sentence.lemma_)
            if count > best_count:
                best_count = count
                best_sentence = sentence

        return best_sentence.text if best_sentence else "No relevant answer found."
    except Exception as e:
        return f"Error in processing the question: {e}"

# ---------------- Summarizer ----------------
def get_data(text, question, answer, word_limit=200):
    global corpus
    corpus = []

    sent_tokens = split_into_sentences(text)
    sent_tokens = [s for s in sent_tokens if s.strip() != ""]
    sent_tokens = list(dict.fromkeys(sent_tokens))  # remove duplicates

    h = numeric_data(sent_tokens)

    # Sort by score
    sorted_sents = sorted(h.items(), key=lambda x: x[1], reverse=True)

    summary = ""
    total_words = 0
    for sentence, score in sorted_sents:
        words_in_sentence = len(sentence.split())
        if total_words + words_in_sentence <= word_limit:
            summary += " " + sentence
            total_words += words_in_sentence
        else:
            break
    return summary.strip()

# ---------------- Streamlit App ----------------
def main():
    st.title("📄 PDF Summarizer & QnA with NLP")

    uploaded_pdf = st.file_uploader("Upload your PDF file", type=["pdf"])
    word_limit = st.number_input("Word limit for summary", min_value=20, max_value=500, value=100)

    if uploaded_pdf:
        with st.spinner("Extracting text..."):
            text = extract_text_from_pdf(uploaded_pdf)

        if st.button("Generate Summary"):
            summary = get_data(text, "", 0, word_limit)
            st.subheader("📌 Summary")
            st.write(summary)

        st.subheader("Ask a Question")
        question = st.text_input("Enter your question based on the PDF")
        if st.button("Get Answer"):
            answer = answer_question(text, question)
            st.success(answer)

if __name__ == "__main__":
    main()
