import streamlit as st
import os
import traceback
import fitz  # PyMuPDF for PDF text extraction
from langchain.chains.llm import LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
import sqlite3
import hashlib

# Set Hugging Face token
HF_TOKEN = os.environ.get("HF_TOKEN")

# Path to FAISS vector store
DB_FAISS_PATH = 'vectorstore/db_faiss'

# SQLite Database Setup
def init_db():
    conn = sqlite3.connect('users.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    ''')
    conn.commit()
    return conn

# Hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Register a new user
def register_user(conn, username, password):
    c = conn.cursor()
    password_hash = hash_password(password)
    try:
        c.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)', (username, password_hash))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        st.error("Username already exists. Please choose a different username.")
        return False

# Authenticate a user
def authenticate_user(conn, username, password):
    c = conn.cursor()
    password_hash = hash_password(password)
    c.execute('SELECT * FROM users WHERE username = ? AND password_hash = ?', (username, password_hash))
    user = c.fetchone()
    return user is not None

# User Authentication UI on the Main Page
def auth_ui(conn):
    st.title("User Authentication")
    choice = st.radio("Choose an option", ["Login", "Sign Up"])

    if choice == "Login":
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if authenticate_user(conn, username, password):
                st.session_state['authenticated'] = True
                st.session_state['username'] = username
                st.success("Logged in successfully!")
                st.rerun()  # Refresh the page to show the main app
            else:
                st.error("Invalid username or password.")

    elif choice == "Sign Up":
        st.subheader("Sign Up")
        new_username = st.text_input("Choose a username")
        new_password = st.text_input("Choose a password", type="password")
        confirm_password = st.text_input("Confirm password", type="password")
        if st.button("Sign Up"):
            if new_password == confirm_password:
                if register_user(conn, new_username, new_password):
                    st.success("Account created successfully! Please log in.")
            else:
                st.error("Passwords do not match.")

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}  # Use "cuda" if you have a GPU
    )
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm(huggingface_repo_id, HF_TOKEN=HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        max_length=512,
        token=HF_TOKEN
    )
    return llm

def extract_text_from_pdf(uploaded_file):
    """Extract text from uploaded PDF file."""
    text = ""
    try:
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text("text") + "\n"
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
    return text.strip()

def summarize_text(text, llm):
    """Generate a concise and readable summary of extracted text using an LLM."""
    summary_prompt = """
    Summarize the given medical report in simple language. Focus on the following key points:
    - Patient symptoms
    - Diagnostic test results
    - Diagnosis        
    - Treatment recommendations or next steps

    Keep the summary concise and under 300 words. Use bullet points for better readability.

    Report:
    {report}

    Summary:
    """
    prompt_template = PromptTemplate(template=summary_prompt, input_variables=["report"])
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)
    
    try:
        response = llm_chain.invoke({"report": text})
        summary = response.get("text", "No summary generated.")
        
        # Ensure the summary uses bullet points and is well-formatted
        if "-" not in summary:  # If the LLM didn't use bullet points, format it
            summary = "- " + summary.replace("\n", "\n- ")
        
        return summary
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return ""

def main():
    # Initialize SQLite database
    conn = init_db()

    # Set Streamlit UI
    st.set_page_config(page_title="Ask Chatbot", page_icon="🤖", layout="centered", initial_sidebar_state="auto")

    # User Authentication
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False

    if not st.session_state['authenticated']:
        auth_ui(conn)
        st.stop()  # Stop further execution if the user is not authenticated

    # Display username and logout button
    st.write(f"Logged in as: **{st.session_state['username']}**")
    if st.button("Logout"):
        st.session_state['authenticated'] = False
        st.session_state.pop('username', None)
        st.rerun()  # Refresh the page to show the login screen

    # Main App: Chatbot and PDF Upload
    st.title("Medibot : Get the Relevant Medical Info")

    # Chatbot UI
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Ask a medical question...")

    # PDF Upload Section
    uploaded_file = st.file_uploader("Upload a medical report (PDF)", type=["pdf"])

    if uploaded_file:
        with st.spinner("Extracting text from PDF..."):
            extracted_text = extract_text_from_pdf(uploaded_file)

        if extracted_text:
            st.subheader("Extracted Text:")
            st.text_area("Raw Text", extracted_text, height=200)

            # Load LLM for summarization
            HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
            llm = load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN)

            with st.spinner("Generating Summary..."):
                summary = summarize_text(extracted_text, llm)

            st.subheader("Summary:")
            st.markdown(summary)  # Use markdown for better formatting

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer the user's question.
        If you don’t know the answer, just say that you don’t know. Don't try to make up an answer.
        Only provide information from the given context.

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk.
        """

        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        llm = load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN)

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            prompt_template = set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)
            llm_chain = LLMChain(llm=llm, prompt=prompt_template)

            # Create the StuffDocumentsChain
            combine_documents_chain = StuffDocumentsChain(
                llm_chain=llm_chain,
                document_variable_name="context"
            )

            # Create the QA chain
            qa_chain = RetrievalQA(
                combine_documents_chain=combine_documents_chain,
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True
            )

            response = qa_chain.invoke({'query': prompt})
            result = response["result"]
            result_to_show = result

            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main()