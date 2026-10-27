
import os
import re
import io

import pdfplumber
import streamlit as st
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

st.set_page_config(page_title="Insurance RAG Assistant")
st.title("Insurance Policy RAG Assistant")
st.caption("Ask questions grounded in your uploaded insurance documents.")

# Configuration
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 4
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gemini-2.5-flash"


def extract_pdf(uploaded_file):
    """Extract page text and tables, preserving page metadata."""
    documents = []
    pdf_bytes = uploaded_file.getvalue()

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""

            # Extract tables separately and retain row/column relationships.
            table_text = []
            for table in page.extract_tables() or []:
                for row in table:
                    if row:
                        cells = [
                            re.sub(r"\s+", " ", str(cell)).strip()
                            if cell is not None else ""
                            for cell in row
                        ]
                        if any(cells):
                            table_text.append(" | ".join(cells))

            combined = text.strip()
            if table_text:
                combined += "\n\nExtracted tables:\n"
                combined += "\n".join(table_text)

            if combined.strip():
                documents.append(
                    Document(
                        page_content=combined,
                        metadata={
                            "source": uploaded_file.name,
                            "page": page_number,
                        },
                    )
                )

    if not documents:
        raise ValueError(
            "No extractable text found. The PDF may be scanned "
            "and require OCR."
        )

    return documents


def build_vector_store(documents):
    """Create chunks and index them for semantic retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No chunks were created from the PDF.")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    return FAISS.from_documents(chunks, embeddings), len(chunks)


def answer_question(vector_store, question, llm):
    """Retrieve evidence and generate a grounded answer."""
    retriever = vector_store.as_retriever(
        search_kwargs={"k": TOP_K}
    )
    retrieved_docs = retriever.invoke(question)

    if not retrieved_docs:
        return "I could not find relevant evidence.", []

    context_parts = []
    for doc in retrieved_docs:
        context_parts.append(
            f"Source: {doc.metadata['source']}, "
            f"page: {doc.metadata['page']}\n"
            f"{doc.page_content}"
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""
You are an insurance policy assistant.

Answer the user's question using only the evidence below.

Rules:
- Do not invent coverage, exclusions, amounts, or conditions.
- If the evidence is insufficient, explicitly say so.
- Distinguish policy wording from your interpretation.
- Mention relevant conditions and exclusions.
- Cite the source filename and page number for factual claims.
- If sources conflict, explain the conflict rather than guessing.

Evidence:
{context}

Question:
{question}

Answer with supporting source citations:
"""

    response = llm.invoke(prompt)
    return response.content, retrieved_docs


# API key validation
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error(
        "GOOGLE_API_KEY is missing. Add it to your .env file "
        "and restart Streamlit."
    )
    st.stop()

# Initialise models once per Streamlit session.
@st.cache_resource
def load_models():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0,
        google_api_key=api_key,
    )
    return embeddings, llm


embeddings, llm = load_models()

uploaded_files = st.file_uploader(
    "Upload insurance policy PDFs",
    type=["pdf"],
    accept_multiple_files=True,
)

if st.button("Process documents"):
    if not uploaded_files:
        st.warning("Please upload at least one PDF.")
    else:
        try:
            all_documents = []
            for uploaded_file in uploaded_files:
                all_documents.extend(extract_pdf(uploaded_file))

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            chunks = splitter.split_documents(all_documents)

            if not chunks:
                raise ValueError("No searchable chunks were created.")

            vector_store = FAISS.from_documents(chunks, embeddings)

            st.session_state["vector_store"] = vector_store
            st.session_state["processed_files"] = [
                f.name for f in uploaded_files
            ]

            st.success(
                f"Processed {len(uploaded_files)} PDF(s), "
                f"{len(all_documents)} page records and "
                f"{len(chunks)} chunks."
            )

        except Exception as exc:
            st.error(f"Document processing failed: {exc}")

if "vector_store" in st.session_state:
    st.subheader("Ask a question")

    question = st.text_input(
        "For example: What exclusions apply to water damage?"
    )

    if st.button("Get answer"):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            try:
                with st.spinner("Searching policy evidence..."):
                    answer, sources = answer_question(
                        st.session_state["vector_store"],
                        question,
                        llm,
                    )

                st.markdown("### Answer")
                st.write(answer)

                with st.expander("Inspect retrieved evidence"):
                    for doc in sources:
                        st.markdown(
                            f"**{doc.metadata['source']} — "
                            f"Page {doc.metadata['page']}**"
                        )
                        st.write(doc.page_content)

            except Exception as exc:
                st.error(f"Question processing failed: {exc}")
