"""Streamlit interface: pick a policy, describe a claim, get a verdict with its clauses."""
import os

import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from rag import (EMBEDDING_MODEL, EXAMPLE_CLAIMS, LLM_MODEL, PUBLIC_POLICIES, answer_claim,
                 build_index, download_public_policy, extract_policy, is_verbatim)

load_dotenv()

st.set_page_config(page_title="Claim Coverage Checker", page_icon="🏠")
st.title("Is this claim covered?")
st.caption("Answers come only from the selected policy, with the clauses quoted word for word. "
           "When the policy does not settle the question, the answer is UNCLEAR.")

api_key = os.getenv("GOOGLE_API_KEY")  # Streamlit Cloud exposes Secrets as env variables too
if not api_key:
    st.error("GOOGLE_API_KEY is missing. Add it to your .env file (or Streamlit Secrets) and restart.")
    st.stop()


@st.cache_resource
def load_models():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0, google_api_key=api_key)
    return embeddings, llm


@st.cache_resource(show_spinner="Downloading and indexing the public policy...")
def load_public_indexes(_embeddings):
    indexes = {}
    for name in PUBLIC_POLICIES:
        try:
            indexes[name] = build_index(extract_policy(download_public_policy(name), name), _embeddings)
        except Exception as exc:
            st.warning(f"Could not load {name}: {exc}")
    return indexes


embeddings, llm = load_models()
if "indexes" not in st.session_state:
    st.session_state["indexes"] = dict(load_public_indexes(embeddings))

with st.expander("Add your own policy PDF"):
    uploaded_files = st.file_uploader("Insurance policy PDFs", type=["pdf"], accept_multiple_files=True)
    if st.button("Process documents"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF.")
        for f in uploaded_files or []:
            try:
                st.session_state["indexes"][f.name] = build_index(
                    extract_policy(f.getvalue(), f.name), embeddings)
                st.success(f"Processed {f.name}.")
            except Exception as exc:
                st.error(f"{f.name}: {exc}")

indexes = st.session_state["indexes"]
if not indexes:
    st.info("No policy loaded yet. Upload a policy PDF above.")
    st.stop()

policy = st.selectbox("Policy", list(indexes))
if policy in PUBLIC_POLICIES:
    st.caption(f"Source: [official policy wording]({PUBLIC_POLICIES[policy]['url']})")

choice = st.radio("Try an example or write your own", EXAMPLE_CLAIMS + ["Write my own"])
claim = st.text_area("Claim", value="" if choice == "Write my own" else choice, height=90)

if st.button("Check coverage", type="primary"):
    if not claim.strip():
        st.warning("Enter a claim first.")
        st.stop()
    try:
        with st.spinner("Searching the policy..."):
            result, evidence = answer_claim(indexes[policy], policy, claim, llm)
    except Exception as exc:
        st.error(f"Could not check the claim: {exc}")
        st.stop()

    colour = {"COVERED": "green", "NOT COVERED": "red", "UNCLEAR": "orange"}[result.verdict]
    st.markdown(f"## :{colour}[{result.verdict}]")
    st.write(result.reasoning)
    if result.assumptions:
        st.warning("**Assumptions behind this answer:**\n" +
                   "\n".join(f"- {a}" for a in result.assumptions))
    if result.missing_info:
        st.info(f"**What would change the answer:** {result.missing_info}")

    st.subheader("Clauses relied on")
    for c in result.citations:
        check = ("✅ found in policy" if is_verbatim(c.quote, evidence)
                 else "⚠️ not found word for word, check manually")
        st.markdown(f"> {c.quote}\n\nPage {c.page} · {check}")

    with st.expander(f"All {len(evidence)} extracts the model saw"):
        for d in evidence:
            st.markdown(f"**Page {d.metadata['page']} · {d.metadata['section']}**")
            st.text(d.page_content)
