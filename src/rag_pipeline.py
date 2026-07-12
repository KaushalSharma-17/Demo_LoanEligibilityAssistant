"""
RAG Pipeline for Loan Policy Q&A.
Loads policy documents, builds a FAISS vector store, and answers questions
using Claude Haiku via RetrievalQA. Embeddings use FakeEmbeddings (no OpenAI key needed).
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import FakeEmbeddings
# from langchain_openai import OpenAIEmbeddings  # commented out — not using OpenAI

load_dotenv()
logger = logging.getLogger(__name__)

FAISS_INDEX_PATH = "faiss_index"
DATA_DIR = "data"

RAG_PROMPT_TEMPLATE = """You are a professional bank loan assistant. Use the following context 
extracted from official bank policy documents to answer the customer's question accurately.

Context:
{context}

Customer Question: {question}

Instructions:
- Answer ONLY based on the provided context.
- If the answer is not in the context, respond with: "I don't have that information in the current policy documents."
- Be concise, professional, and helpful.
- Use Indian Rupee (₹) notation where applicable.

Answer:"""


def load_documents(data_dir: str = DATA_DIR) -> list:
    """Load all PDF and TXT documents from the data directory."""
    documents = []
    data_path = Path(data_dir)

    if not data_path.exists():
        logger.warning(f"Data directory '{data_dir}' not found.")
        return documents

    for file_path in data_path.iterdir():
        try:
            if file_path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(file_path))
                documents.extend(loader.load())
                logger.info(f"Loaded PDF: {file_path.name}")
            elif file_path.suffix.lower() == ".txt":
                loader = TextLoader(str(file_path), encoding="utf-8")
                documents.extend(loader.load())
                logger.info(f"Loaded TXT: {file_path.name}")
        except Exception as e:
            logger.error(f"Failed to load {file_path.name}: {e}")

    logger.info(f"Total documents loaded: {len(documents)}")
    return documents


def build_vector_store(documents: list, embeddings: FakeEmbeddings) -> FAISS:
    """Split documents and build a FAISS vector store."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    logger.info(f"Total chunks created: {len(chunks)}")

    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(FAISS_INDEX_PATH)
    logger.info(f"FAISS index saved to '{FAISS_INDEX_PATH}'")
    return vector_store


def load_or_build_vector_store(embeddings: FakeEmbeddings, force_rebuild: bool = False) -> FAISS:
    """Load existing FAISS index or build a new one from documents."""
    if not force_rebuild and Path(FAISS_INDEX_PATH).exists():
        logger.info("Loading existing FAISS index.")
        return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

    logger.info("Building new FAISS index from documents.")
    documents = load_documents()
    if not documents:
        raise ValueError("No documents found in the data directory. Please add policy files.")
    return build_vector_store(documents, embeddings)


def build_rag_chain(force_rebuild: bool = False) -> RetrievalQA:
    """Build and return the RetrievalQA chain using Claude Haiku. Uses FakeEmbeddings (no OpenAI key needed)."""
    # OpenAI embeddings commented out — using FakeEmbeddings instead
    # embeddings = OpenAIEmbeddings(
    #     model="text-embedding-3-small",
    #     openai_api_key=os.getenv("OPENAI_API_KEY"),
    # )
    embeddings = FakeEmbeddings(size=1536)

    llm = ChatAnthropic(
        model=os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        base_url=os.getenv("ANTHROPIC_BASE_URL"),
        temperature=0,
    )

    vector_store = load_or_build_vector_store(embeddings, force_rebuild=force_rebuild)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    prompt = PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    logger.info("RAG chain built successfully.")
    return chain


def get_document_count(data_dir: str = DATA_DIR) -> int:
    """Return the number of policy documents in the data directory."""
    data_path = Path(data_dir)
    if not data_path.exists():
        return 0
    return sum(1 for f in data_path.iterdir() if f.suffix.lower() in {".pdf", ".txt"})
