import streamlit as st
import fitz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq


# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------

st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="📄",
    layout="wide"
)


# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("📄 HR Policy Assistant")
st.write(
    "Upload an HR Policy PDF and ask questions about its contents."
)


# ---------------------------------------------------
# LOAD EMBEDDING MODEL
# ---------------------------------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()


# ---------------------------------------------------
# EXTRACT TEXT FROM PDF
# ---------------------------------------------------

def extract_text_from_pdf(pdf_file):

    pdf_bytes = pdf_file.read()

    document = fitz.open(stream=pdf_bytes, filetype="pdf")

    text = ""

    for page in document:
        text += page.get_text()

    document.close()

    return text


# ---------------------------------------------------
# SPLIT TEXT INTO CHUNKS
# ---------------------------------------------------

def split_text(text, chunk_size=500, overlap=100):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ---------------------------------------------------
# CREATE FAISS INDEX
# ---------------------------------------------------

def create_faiss_index(chunks):

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True
    )

    embeddings = embeddings.astype("float32")

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


# ---------------------------------------------------
# RETRIEVE RELEVANT CHUNKS
# ---------------------------------------------------

def retrieve_chunks(question, chunks, index, top_k=4):

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    )

    question_embedding = question_embedding.astype("float32")

    faiss.normalize_L2(question_embedding)

    scores, indices = index.search(
        question_embedding,
        top_k
    )

    retrieved_chunks = []

    for i in indices[0]:

        if i != -1:
            retrieved_chunks.append(chunks[i])

    return retrieved_chunks


# ---------------------------------------------------
# GENERATE ANSWER USING GROQ
# ---------------------------------------------------

def generate_answer(question, context, api_key):

    client = Groq(api_key=api_key)

    prompt = f"""
You are an HR Policy Assistant.

Answer the user's question using ONLY the information
provided in the HR policy context below.

If the answer cannot be found in the context, clearly say:

"I could not find this information in the uploaded HR policy."

Do not make up information.

HR POLICY CONTEXT:
{context}

USER QUESTION:
{question}

Provide a clear and concise answer.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": "You answer questions using the provided HR policy context."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content


# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

with st.sidebar:

    st.header("⚙️ Settings")

    groq_api_key = st.text_input(
        "Enter Groq API Key",
        type="password"
    )

    st.info(
        "Your API key is used to communicate with Groq."
    )


# ---------------------------------------------------
# PDF UPLOAD
# ---------------------------------------------------

uploaded_file = st.file_uploader(
    "📤 Upload your HR Policy PDF",
    type=["pdf"]
)


# ---------------------------------------------------
# PROCESS PDF
# ---------------------------------------------------

if uploaded_file is not None:

    with st.spinner("Reading and processing the HR policy..."):

        try:

            text = extract_text_from_pdf(uploaded_file)

            if not text.strip():

                st.error(
                    "No readable text was found in the PDF."
                )

            else:

                chunks = split_text(text)

                index = create_faiss_index(chunks)

                st.success(
                    f"PDF processed successfully! "
                    f"{len(chunks)} text chunks created."
                )

                st.session_state["chunks"] = chunks
                st.session_state["index"] = index

        except Exception as e:

            st.error(
                f"Error processing PDF: {str(e)}"
            )


# ---------------------------------------------------
# QUESTION INPUT
# ---------------------------------------------------

if (
    "chunks" in st.session_state
    and "index" in st.session_state
):

    st.subheader("💬 Ask a Question")

    question = st.text_input(
        "Enter your question about the HR policy:"
    )

    if st.button("🔍 Ask"):

        if not groq_api_key:

            st.warning(
                "Please enter your Groq API key in the sidebar."
            )

        elif not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner("Searching the HR policy..."):

                relevant_chunks = retrieve_chunks(
                    question,
                    st.session_state["chunks"],
                    st.session_state["index"]
                )

                context = "\n\n".join(
                    relevant_chunks
                )

            with st.spinner("Generating answer..."):

                try:

                    answer = generate_answer(
                        question,
                        context,
                        groq_api_key
                    )

                    st.subheader("🤖 Answer")

                    st.write(answer)

                    with st.expander(
                        "📚 View Retrieved Policy Sections"
                    ):

                        for i, chunk in enumerate(
                            relevant_chunks,
                            start=1
                        ):

                            st.markdown(
                                f"**Section {i}**"
                            )

                            st.write(chunk)

                except Exception as e:

                    st.error(
                        f"Error generating answer: {str(e)}"
                    )
