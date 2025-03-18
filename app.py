import streamlit as st
from streamlit_extras.stylable_container import stylable_container
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # Handle NoneType
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# Function to create a vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to create a conversational chain
def get_conversational_chain(vectorstore):
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not available, respond with:
    'Answer is not available in the context'.\n\n
    Context:\n {context}\n
    Question:\n {question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    return ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

# Function to handle user input and generate responses
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = get_conversational_chain(new_db)

    response = st.session_state.conversation(
        {"question": user_question, "chat_history": st.session_state.chat_history},
        return_only_outputs=True
    )

    # Store chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.chat_history.append({"user": user_question, "bot": response["answer"]})

# Main function to run the Streamlit app
def main():
    st.markdown(
        """
        <h1 style="display: flex; align-items: center;">
            Chat with PDF 
            <img src="https://img.icons8.com/?size=100&id=80399&format=png&color=000000" width="50" style="margin-left: 3px;">
        </h1>
        """,
        unsafe_allow_html=True
    )

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "user_question" not in st.session_state:
        st.session_state.user_question = ""

    # Layout for text input with an arrow button
    col1, col2 = st.columns([6, 1])

    with col1:
        user_question = st.text_area(
            "Ask a Question from the PDF Files",
            value=st.session_state.user_question,
            key="user_question_input",
            height=80,
            label_visibility="collapsed"
        )

    with col2:
        if st.button("➜"):
            if user_question.strip():
                user_input(user_question)
                st.session_state.user_question = ""  # Reset after processing
                st.rerun()

    # Display chat history in reverse order
    for chat in reversed(st.session_state.chat_history):
        st.write(f"**You:** {chat['user']}")
        st.write(f"**Bot:** {chat['bot']}")
        st.write("---")

    # Sidebar for uploading PDFs
    with st.sidebar:
        st.title("Upload Your Documents")
        pdf_docs = st.file_uploader("Upload", accept_multiple_files=True)
        if st.button("Submit"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

# Run the app
if __name__ == "__main__":
    main()
