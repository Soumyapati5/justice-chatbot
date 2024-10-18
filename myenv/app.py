import os
import streamlit as st
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import faiss
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from groq import Groq

# Load environment variables
load_dotenv()

# Check and download NLTK data only if not available
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
if not os.path.exists(nltk_data_path):
    nltk.download('stopwords', download_dir=nltk_data_path)
    nltk.download('punkt', download_dir=nltk_data_path)
else:
    nltk.data.path.append(nltk_data_path)

# Initialize variables and models
groq_api_key = os.getenv('GROQ_API_KEY')
client = Groq(api_key=groq_api_key)
MODEL = 'llama3-70b-8192'

index_path = os.path.join("embdedDoc/faiss_index.bin")
docstore_path = os.path.join("embdedDoc/docstore.pkl")
index_to_docstore_id_path = os.path.join("embdedDoc/index_to_docstore_id.pkl")

index = faiss.read_index(index_path)
with open(docstore_path, 'rb') as f:
    docstore = pickle.load(f)
with open(index_to_docstore_id_path, 'rb') as f:
    index_to_docstore_id = pickle.load(f)

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS(
    embedding_function=embedding_function,
    index=index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id
)

# Utility functions
def remove_stop_words_and_wh(question):
    stop_words = set(stopwords.words('english'))
    wh_words = {"what", "when", "where", "who", "whom", "which", "why", "how"}
    word_tokens = word_tokenize(question.lower())
    filtered_question = [word for word in word_tokens if word not in stop_words and word not in wh_words]
    return " ".join(filtered_question)

def get_relevant_context(question):
    refined_question = remove_stop_words_and_wh(question)
    results = db.similarity_search_with_score(refined_question, k=5)
    context = " ".join([result[0].page_content for result in results])
    return context

def get_response(user_question):
    system_prompt = """
    You are an AI that answers questions based on given context. 
    Make sure your answers are concise, accurate, and based solely on the provided context.
    """

    messages = [
        {"role": "system", "content": system_prompt},
    ]

    context = get_relevant_context(user_question)
    user_prompt = "Context: " + context + "\n Question: " + user_question + "?"

    local_messages = messages + [{"role": "user", "content": user_prompt}]
    response = client.chat.completions.create(
        model=MODEL,
        messages=local_messages,
        max_tokens=8192,
        top_p=1,
        temperature=0
    )
    response_message = response.choices[0].message
    return response_message.content

# Streamlit interface
st.title("Contextual AI Question Answering")
st.write("Ask a question, and the AI will provide an answer based on relevant context.")

# Input form with Enter key triggering the submission
with st.form(key='question_form'):
    user_question = st.text_input("Enter your question:")
    submit_button = st.form_submit_button("Get Answer")  # Triggers on Enter key or button click

if submit_button:
    if user_question:
        answer = get_response(user_question)
        st.write("**Question:**", user_question)
        st.write("**Answer:**", answer)
    else:
        st.write("Please enter a question to get an answer.")
