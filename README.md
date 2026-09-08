# 📄 HR Policy Assistant

An AI-powered HR Policy Assistant built using Retrieval-Augmented Generation (RAG).

Users can upload an HR Policy PDF and ask questions about the policy.

## 🚀 Features

- Upload HR Policy PDF
- Extract text from PDF
- Split text into smaller chunks
- Generate embeddings using Sentence Transformers
- Store and search embeddings using FAISS
- Retrieve relevant policy sections
- Generate answers using Groq
- Simple Streamlit interface

## 🧠 Technologies Used

- Python
- Streamlit
- FAISS
- Sentence Transformers
- PyMuPDF
- Groq
- NumPy

## 🔄 RAG Workflow

1. User uploads an HR Policy PDF.
2. PyMuPDF extracts the text.
3. The text is divided into chunks.
4. Sentence Transformers converts chunks into embeddings.
5. FAISS stores the embeddings.
6. User asks a question.
7. The question is converted into an embedding.
8. FAISS retrieves the most relevant chunks.
9. Retrieved policy content is sent to Groq.
10. Groq generates an answer based on the policy.

## ▶️ How to Use

1. Open the deployed Streamlit application.
2. Upload an HR Policy PDF.
3. Enter your Groq API key.
4. Ask a question about the policy.
5. View the generated answer.

## ⚠️ Important

The application is designed to answer questions using the uploaded HR policy.

If the requested information is not found in the retrieved policy content, the assistant should say that the information could not be found.

## 👩‍💻 Project

HR Policy Assistant using Retrieval-Augmented Generation (RAG).
