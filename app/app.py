import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load API key from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Page config
st.set_page_config(page_title="Document Chatbot", page_icon="📄")
st.title("📄 Document Chatbot")
st.write("Ask any question about the uploaded document.")


# Load and process the PDF
@st.cache_resource
def load_document(pdf_path):

    # Step 1 — Load the PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    st.write(f"Loaded {len(documents)} pages")

    # Step 2 — Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    st.write(f"Split into {len(chunks)} chunks")

    # Step 3 — Create embeddings and store in ChromaDB
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    st.write(f"Stored in vector database")

    return vectorstore


# Build the QA chain
def build_qa_chain(vectorstore):

    prompt_template = """
    You are a helpful assistant that answers questions based strictly on 
    the provided context. If the answer is not in the context, say 
    "I don't have enough information in the document to answer that."
    
    Context: {context}
    
    Question: {question}
    
    Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        api_key=openai_api_key
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


# Main app
def main():

    # Page header
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # PDF upload
    st.subheader("Upload a Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:

        # Save uploaded file to data folder
        data_dir = os.path.join(base_dir, 'data')
        pdf_path = os.path.join(data_dir, uploaded_file.name)

        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"Uploaded: {uploaded_file.name}")

        # Load and process document
        with st.spinner("Processing document — this may take a minute..."):
            vectorstore = load_document(pdf_path)

        # Build QA chain
        qa_chain = build_qa_chain(vectorstore)

        # Chat interface
        st.subheader("Ask a question")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if question := st.chat_input("Ask something about the document..."):

            st.session_state.messages.append({
                "role": "user",
                "content": question
            })

            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = qa_chain.invoke(question)
                    st.write(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

    else:
        st.info("Please upload a PDF to get started.")


if __name__ == "__main__":
    main()