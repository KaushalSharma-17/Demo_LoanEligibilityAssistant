"""
Streamlit Frontend for the Intelligent Loan Eligibility Assistant.
Tab 1: Policy Q&A chatbot (RAG)
Tab 2: Eligibility Checker form
"""

import json
import logging
import os

import streamlit as st
from dotenv import load_dotenv

from src.agent import build_agent
from src.eligibility_checker import get_raw_eligibility_result
from src.rag_pipeline import build_rag_chain, get_document_count

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Loan Eligibility Assistant",
    page_icon="🏦",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading AI agent...")
def get_agent():
    """Initialize and cache the LangChain agent."""
    return build_agent()


@st.cache_resource(show_spinner="Loading RAG pipeline...")
def get_rag_chain():
    """Initialize and cache the RAG chain."""
    return build_rag_chain()


def reload_resources():
    """Clear cached resources and rebuild."""
    st.cache_resource.clear()
    st.rerun()


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏦 Loan Assistant")
    st.divider()

    st.markdown("**Model**")
    st.code(os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307"))

    st.markdown("**Embeddings**")
    st.code("text-embedding-3-small")

    doc_count = get_document_count()
    st.markdown("**Policy Documents Loaded**")
    st.metric(label="", value=doc_count)

    if st.button("🔄 Reload Documents", use_container_width=True):
        reload_resources()

    st.divider()
    st.caption("Powered by LangChain + Claude Haiku")


# ── Main Tabs ─────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["💬 Policy Assistant", "✅ Eligibility Checker"])


# ── Tab 1: Policy Q&A ─────────────────────────────────────────────────────────
with tab1:
    st.header("Loan Policy Assistant")
    st.caption("Ask any question about our loan products, interest rates, required documents, or policies.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about loan policies..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching policy documents..."):
                try:
                    rag_chain = get_rag_chain()
                    result = rag_chain.invoke({"query": prompt})
                    answer = result.get("result", "Sorry, I could not find an answer.")
                    source_docs = result.get("source_documents", [])

                    st.markdown(answer)

                    if source_docs:
                        with st.expander(f"📄 Sources ({len(source_docs)} chunks used)"):
                            for i, doc in enumerate(source_docs, 1):
                                st.markdown(f"**Chunk {i}**")
                                st.text(doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content)
                                st.divider()

                except Exception as e:
                    answer = f"⚠️ Error: {e}"
                    st.error(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()

    st.divider()
    st.markdown("**Sample questions to try:**")
    cols = st.columns(2)
    with cols[0]:
        st.info("What is the minimum salary for a personal loan?")
        st.info("What credit score is needed for a home loan?")
    with cols[1]:
        st.info("What documents are required?")
        st.info("What is the maximum loan amount I can get?")


# ── Tab 2: Eligibility Checker ────────────────────────────────────────────────
with tab2:
    st.header("Loan Eligibility Checker")
    st.caption("Fill in your details to check if you qualify for a loan.")

    with st.form("eligibility_form"):
        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input("Full Name", placeholder="e.g. John Doe")
            age = st.number_input("Age (years)", min_value=18, max_value=80, value=30)
            salary = st.number_input("Monthly Salary (₹)", min_value=0, value=50000, step=1000)

        with col2:
            loan_amount = st.number_input("Loan Amount Requested (₹)", min_value=0, value=300000, step=10000)
            credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=720)

        submitted = st.form_submit_button("🔍 Check Eligibility", use_container_width=True)

    if submitted:
        if not name.strip():
            st.warning("Please enter the applicant's name.")
        else:
            input_json = json.dumps({
                "name": name,
                "age": int(age),
                "salary": int(salary),
                "loan_amount": int(loan_amount),
                "credit_score": int(credit_score),
            })

            with st.spinner("Evaluating eligibility..."):
                try:
                    result = get_raw_eligibility_result(input_json)

                    st.divider()

                    if result.eligible:
                        st.success(f"## ✅ {result.verdict}")
                        st.balloons()
                    else:
                        st.error(f"## ❌ {result.verdict}")

                    st.markdown("### Eligibility Report")
                    st.markdown(result.formatted_response)

                    st.markdown("### Rule Breakdown")
                    for rule in result.rule_results:
                        icon = "✅" if rule["passed"] else "❌"
                        color = "green" if rule["passed"] else "red"
                        st.markdown(
                            f":{color}[{icon} **{rule['rule']}** — {rule['message']}]"
                        )

                    if not result.eligible:
                        st.divider()
                        st.markdown("### 💡 Personalized Recommendations")
                        with st.spinner("Generating recommendations..."):
                            from src.recommendation_engine import generate_recommendations
                            failed_rules = [r for r in result.rule_results if not r["passed"]]
                            recommendations = generate_recommendations(result.applicant, failed_rules)
                            st.info(recommendations)

                except Exception as e:
                    st.error(f"⚠️ An error occurred: {e}")
                    logger.error(f"Eligibility check UI error: {e}")
