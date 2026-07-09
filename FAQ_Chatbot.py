!pip install nltk scikit-learn gradio

import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import gradio as gr

faq_questions = [
    "What is Artificial Intelligence?",
    "What is Machine Learning?",
    "What is Python?",
    "What is the duration of the internship?",
    "Will I receive a certificate?",
    "How can I contact support?",
    "What is Data Science?",
    "What are the internship requirements?"
]

faq_answers = [
    "Artificial Intelligence is the simulation of human intelligence by machines.",
    "Machine Learning is a branch of AI that enables computers to learn from data.",
    "Python is a popular programming language used in AI, web development, and data science.",
    "The internship duration is one month.",
    "Yes, you will receive a certificate after successfully completing the internship.",
    "You can contact support through the official website or email.",
    "Data Science is the process of extracting useful insights from data.",
    "You need to complete the assigned tasks and submit them before the deadline."
]

import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    words = word_tokenize(text)

    filtered_words = []

    for word in words:
        if word.isalnum() and word not in stop_words:
            filtered_words.append(word)

    return " ".join(filtered_words)

processed_questions = []

for question in faq_questions:
    processed_questions.append(preprocess(question))

print(processed_questions)

vectorizer = TfidfVectorizer()

faq_vectors = vectorizer.fit_transform(processed_questions)

print("TF-IDF vectors created successfully!")

def chatbot(user_question):
    # User question preprocess
    processed_input = preprocess(user_question)

    # Convert user question into TF-IDF vector
    user_vector = vectorizer.transform([processed_input])

    # Calculate similarity
    similarity = cosine_similarity(user_vector, faq_vectors)

    # Find best matching question
    best_match = similarity.argmax()

    # If similarity is very low
    if similarity[0][best_match] < 0.2:
        return "Sorry, I don't have an answer for that question."

    # Return corresponding answer
    return faq_answers[best_match]

app = gr.Interface(
    fn=chatbot,
    inputs=gr.Textbox(
        lines=2,
        placeholder="Ask your question here...",
        label="Your Question"
    ),
    outputs=gr.Textbox(label="Chatbot Answer"),
    title="FAQ Chatbot",
    description="Ask questions about the internship."
)

app.launch()