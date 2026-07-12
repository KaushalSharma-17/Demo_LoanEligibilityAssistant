# 🏦 Intelligent Loan Eligibility Assistant

An AI-powered loan assistant built with **LangChain**, **Claude Haiku**, and **Streamlit**.

---

## Architecture

```
              User (Streamlit UI)
                      |
            ┌─────────┴──────────┐
            │                    │
      Policy Q&A Tab       Eligibility Tab
            │                    │
       RAG Pipeline        Rule Engine
            │                    │
    OpenAI Embeddings     Claude Haiku (LLM)
            │                    │
       FAISS Vector DB    Recommendation Engine
```

---

## Features

| Feature | Description |
|---|---|
| **Policy Q&A (RAG)** | Chat with bank policy documents using FAISS + OpenAI embeddings |
| **Eligibility Checker** | Rule-based check with Claude Haiku-formatted response |
| **Recommendation Engine** | Personalized improvement tips for rejected applicants |
| **Conversation Memory** | Agent remembers context across turns (k=5 window) |

---

## Prerequisites

- Python 3.10+
- Anthropic API key (for Claude Haiku)
- OpenAI API key (for embeddings only)

---

## Installation

```bash
# 1. Clone or navigate to the project folder
cd LoanEligibilityAssistant

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Environment Setup

```bash
# Copy the example env file
copy .env.example .env
```

Edit `.env` and fill in your API keys:

```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
ANTHROPIC_MODEL=claude-3-haiku-20240307
OPENAI_API_KEY=your_openai_api_key_here
```

> **Note**: OpenAI key is used **only** for generating text embeddings (`text-embedding-3-small`).
> All LLM inference uses Claude Haiku via the Anthropic API.

---

## Adding Custom Policy Documents

Place any `.pdf` or `.txt` policy files inside the `data/` folder:

```
data/
├── sample_policy.txt       ← included by default
├── your_custom_policy.pdf  ← add your own
└── credit_guidelines.txt   ← add your own
```

After adding new documents, click **"🔄 Reload Documents"** in the sidebar to rebuild the vector index.

---

## Running the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## Sample Test Cases

### Policy Q&A
| Question | Expected Answer |
|---|---|
| What is the minimum salary for a personal loan? | ₹30,000/month |
| What credit score is needed for a home loan? | 720 minimum |
| What documents are required? | Aadhar, PAN, salary slips, bank statements, Form 16 |
| What is the maximum home loan amount? | ₹1 Crore |

### Eligibility Checker
| Input | Expected Result |
|---|---|
| Age: 30, Salary: ₹60,000, Loan: ₹5,00,000, Score: 750 | ✅ ELIGIBLE |
| Age: 25, Salary: ₹25,000, Loan: ₹3,00,000, Score: 680 | ❌ NOT ELIGIBLE (salary + credit score fail) |
| Age: 19, Salary: ₹40,000, Loan: ₹2,00,000, Score: 710 | ❌ NOT ELIGIBLE (age fails) |
| Age: 35, Salary: ₹50,000, Loan: ₹6,00,000, Score: 700 | ✅ ELIGIBLE |

---

## Project Structure

```
LoanEligibilityAssistant/
├── app.py                        # Streamlit frontend
├── requirements.txt
├── .env.example
├── data/
│   └── sample_policy.txt
├── src/
│   ├── __init__.py
│   ├── rag_pipeline.py           # RAG Q&A pipeline
│   ├── eligibility_checker.py    # Rule engine + LLM formatter
│   ├── recommendation_engine.py  # Personalized recommendations
│   └── agent.py                  # LangChain ReAct Agent
└── faiss_index/                  # Auto-generated on first run
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Anthropic Claude Haiku |
| Framework | LangChain |
| Embeddings | OpenAI text-embedding-3-small |
| Vector Store | FAISS |
| Frontend | Streamlit |
| Backend | Python 3.10+ |
