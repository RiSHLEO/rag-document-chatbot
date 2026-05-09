# RAG Document Chatbot

An AI-powered chatbot that answers questions about any PDF document using 
Retrieval Augmented Generation (RAG). Upload any PDF and ask questions — 
the app retrieves the most relevant sections and generates accurate answers 
grounded strictly in the document.

**Live App:** [Click here to view the app](https://rag-document-chatbot-3xc55appqu9slljfzstbfqt.streamlit.app/)

---

## How It Works

1. **Upload** any PDF document
2. **Chunking** — the document is split into overlapping chunks of ~1000 characters
3. **Embedding** — each chunk is converted to a vector using OpenAI's embedding model
4. **Storage** — vectors are stored in ChromaDB, a local vector database
5. **Retrieval** — when a question is asked, the 3 most semantically similar chunks are retrieved
6. **Generation** — GPT-3.5-turbo generates an answer based strictly on the retrieved chunks
7. **Grounded answers** — if the answer isn't in the document, the app says so rather than hallucinating

---

## Technical Stack

- **LLM:** GPT-3.5-turbo via OpenAI API
- **Framework:** LangChain
- **Vector Database:** ChromaDB
- **Embeddings:** OpenAI text-embedding-ada-002
- **Frontend:** Streamlit
- **Document Processing:** PyPDF

---

## Key Features

- Upload any PDF — not limited to one document
- Answers grounded strictly in the uploaded document
- Refuses to answer questions outside the document scope
- Conversation history maintained across messages
- Processes documents of 100+ pages efficiently

---

## How to Run Locally

```bash
git clone https://github.com/RiSHLEO/rag-document-chatbot
cd rag-document-chatbot
pip install -r requirements.txt
```

Create a `.env` file in the root folder: OPENAI_API_KEY=your-key-here

Then run:
```bash
cd app
streamlit run app.py
```

---

## Example Use Cases

- Chat with a company annual report
- Query a research paper
- Ask questions about a legal document
- Summarise and explore technical documentation

---

## What I Would Improve With More Time

- Support multiple documents simultaneously
- Add source citations showing which page each answer came from
- Add a document summary feature on upload
- Switch to GPT-4 for more complex reasoning tasks
- Add support for other file types — Word documents, web pages
