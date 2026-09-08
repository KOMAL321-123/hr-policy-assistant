```python
import streamlit as st
import fitz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .answer-box {
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📄 HR Policy Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload an HR Policy PDF and ask questions in natural language.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# LOAD SENTENCE TRANSFORMER MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# ============================================================
# GET GROQ API KEY FROM STREAMLIT SECRETS
# ============================================================

try:

    groq_api_key = st.secrets["GROQ_API_KEY"]

except Exception:

    groq_api_key = ""


# ============================================================
# EXTRACT TEXT FROM PDF
# ============================================================

def extract_text_from_pdf(pdf_file):

    pdf_bytes = pdf_file.read()

    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    pages_text = []

    for page_number, page in enumerate(document):

        page_text = page.get_text()

        if page_text.strip():

            pages_text.append(
                f"[Page {page_number + 1}]\n{page_text}"
            )

    document.close()

    return "\n\n".join(pages_text)


# ============================================================
# SPLIT TEXT INTO CHUNKS
# ============================================================

def split_text(
    text,
    chunk_size=450,
    overlap=80
):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(
            words[start:end]
        )

        if chunk.strip():

            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ============================================================
# CREATE FAISS INDEX
# ============================================================

def create_faiss_index(chunks):

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    embeddings = embeddings.astype(
        "float32"
    )

    faiss.normalize_L2(
        embeddings
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    return index


# ============================================================
# RETRIEVE RELEVANT CHUNKS
# ============================================================

def retrieve_chunks(
    question,
    chunks,
    index,
    top_k=5
):

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        show_progress_bar=False
    )

    question_embedding = question_embedding.astype(
        "float32"
    )

    faiss.normalize_L2(
        question_embedding
    )

    scores, indices = index.search(
        question_embedding,
        top_k
    )

    retrieved_chunks = []

    for score, index_number in zip(
        scores[0],
        indices[0]
    ):

        if index_number != -1:

            retrieved_chunks.append(
                {
                    "text": chunks[index_number],
                    "score": float(score)
                }
            )

    return retrieved_chunks


# ============================================================
# GENERATE ANSWER USING GROQ
# ============================================================

def generate_answer(
    question,
    retrieved_chunks,
    api_key
):

    context_parts = []

    for item in retrieved_chunks:

        context_parts.append(
            item["text"]
        )

    context = "\n\n".join(
        context_parts
    )

    client = Groq(
        api_key=api_key
    )

    system_prompt = """
You are a professional HR Policy Assistant.

Your job is to answer questions using ONLY the information
provided in the uploaded HR policy.

IMPORTANT RULES:

1. Do not use outside knowledge.
2. Do not invent or guess information.
3. Answer only what the user's question asks.
4. Keep the answer SHORT and COMPLETE.
5. Prefer 2 to 5 sentences when possible.
6. Remove unnecessary background information.
7. Summarize long policy text into simple language.
8. If the policy contains a specific number, date,
   requirement, eligibility rule, or procedure, include it.
9. If the answer is not available in the provided context,
   say exactly:
   "I could not find this information in the uploaded HR policy."
10. Do not mention that you are an AI unless necessary.
11. Do not repeat the user's question.
12. Use bullet points only when they make the answer clearer.
"""

    user_prompt = f"""
HR POLICY INFORMATION:

{context}

USER QUESTION:

{question}

Answer the question using only the HR policy information above.
Give a concise, clear, and complete answer.
"""

    response = client.chat.completions.create(

        model="openai/gpt-oss-20b",

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],

        temperature=0.1,

        max_tokens=400
    )

    return response.choices[0].message.content


# ============================================================
# SESSION STATE
# ============================================================

if "chunks" not in st.session_state:

    st.session_state.chunks = None


if "index" not in st.session_state:

    st.session_state.index = None


if "chat_history" not in st.session_state:

    st.session_state.chat_history = []


if "file_name" not in st.session_state:

    st.session_state.file_name = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Assistant")

    st.write(
        "Upload an HR policy PDF and ask questions "
        "about its contents."
    )

    if groq_api_key:

        st.success(
            "Groq API connected"
        )

    else:

        st.error(
            "Groq API key is not configured."
        )

    st.divider()

    if st.session_state.chunks:

        st.metric(
            "Policy Chunks",
            len(st.session_state.chunks)
        )

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.chat_history = []

        st.rerun()


# ============================================================
# PDF UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📤 Upload your HR Policy PDF",
    type=["pdf"],
    help="Upload a text-based HR Policy PDF."
)


# ============================================================
# PROCESS PDF
# ============================================================

if uploaded_file is not None:

    if (
        st.session_state.file_name
        != uploaded_file.name
    ):

        with st.spinner(
            "📖 Reading and processing your HR policy..."
        ):

            try:

                text = extract_text_from_pdf(
                    uploaded_file
                )

                if not text.strip():

                    st.error(
                        "No readable text was found in this PDF. "
                        "Please upload a text-based PDF."
                    )

                    st.stop()

                chunks = split_text(
                    text
                )

                if not chunks:

                    st.error(
                        "Could not create text chunks from the PDF."
                    )

                    st.stop()

                index = create_faiss_index(
                    chunks
                )

                st.session_state.chunks = chunks

                st.session_state.index = index

                st.session_state.file_name = (
                    uploaded_file.name
                )

                st.session_state.chat_history = []

                st.success(
                    f"✅ {uploaded_file.name} processed successfully!"
                )

            except Exception as e:

                st.error(
                    f"Error processing PDF: {str(e)}"
                )


# ============================================================
# SHOW CURRENT PDF
# ============================================================

if st.session_state.chunks:

    st.info(
        f"📄 Current policy: "
        f"**{st.session_state.file_name}**"
    )


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.chat_history:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and "sources" in message
        ):

            with st.expander(
                "📚 View Retrieved Policy Sections"
            ):

                for number, source in enumerate(
                    message["sources"],
                    start=1
                ):

                    st.markdown(
                        f"**Policy Section {number}**"
                    )

                    st.write(
                        source["text"]
                    )

                    st.divider()


# ============================================================
# QUESTION INPUT
# ============================================================

if st.session_state.chunks:

    question = st.chat_input(
        "💬 Ask anything about the HR policy..."
    )

    if question:

        # ----------------------------------------------------
        # SHOW USER QUESTION
        # ----------------------------------------------------

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message(
            "user"
        ):

            st.write(
                question
            )


        # ----------------------------------------------------
        # CHECK API KEY
        # ----------------------------------------------------

        if not groq_api_key:

            answer = (
                "Groq API key is not configured. "
                "Please add GROQ_API_KEY to Streamlit Secrets."
            )

            sources = []


        else:

            # ------------------------------------------------
            # RETRIEVE RELEVANT INFORMATION
            # ------------------------------------------------

            with st.spinner(
                "🔎 Searching the HR policy..."
            ):

                try:

                    sources = retrieve_chunks(
                        question,
                        st.session_state.chunks,
                        st.session_state.index,
                        top_k=5
                    )

                except Exception as e:

                    sources = []

                    answer = (
                        f"Error searching the policy: {str(e)}"
                    )


            # ------------------------------------------------
            # GENERATE ANSWER
            # ------------------------------------------------

            if sources:

                with st.spinner(
                    "🤖 Preparing a concise answer..."
                ):

                    try:

                        answer = generate_answer(
                            question,
                            sources,
                            groq_api_key
                        )

                    except Exception as e:

                        answer = (
                            f"Error generating answer: {str(e)}"
                        )

            else:

                answer = (
                    "I could not find relevant information "
                    "in the uploaded HR policy."
                )


        # ----------------------------------------------------
        # SHOW ANSWER
        # ----------------------------------------------------

        with st.chat_message(
            "assistant"
        ):

            st.markdown(
                f'<div class="answer-box">{answer}</div>',
                unsafe_allow_html=True
            )

            if sources:

                with st.expander(
                    "📚 View Retrieved Policy Sections"
                ):

                    for number, source in enumerate(
                        sources,
                        start=1
                    ):

                        st.markdown(
                            f"**Policy Section {number}**"
                        )

                        st.write(
                            source["text"]
                        )

                        st.divider()


        # ----------------------------------------------------
        # SAVE ANSWER TO CHAT HISTORY
        # ----------------------------------------------------

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources
            }
        )


else:

    st.info(
        "👆 Upload an HR Policy PDF above to start asking questions."
    )
```
