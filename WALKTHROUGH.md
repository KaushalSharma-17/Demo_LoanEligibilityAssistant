# 🏦 Loan Eligibility Assistant — Code Walkthrough

---

## 🗂️ Project Structure First

```
LoanEligibilityAssistant/
├── app.py                    ← The UI (what users see)
├── .env                      ← Your secret keys (never shared)
├── .env.example              ← Template showing what keys are needed
├── requirements.txt          ← List of libraries to install
├── data/
│   └── sample_policy.txt     ← Bank policy document (the "knowledge base")
└── src/
    ├── rag_pipeline.py       ← Feature 1: Answer policy questions
    ├── eligibility_checker.py← Feature 2: Check if someone qualifies
    ├── recommendation_engine.py ← Feature 3: Give improvement advice
    └── agent.py              ← The "brain" that connects all 3
```

Think of it like a **bank branch**:
- `sample_policy.txt` = the policy manual on the shelf
- `rag_pipeline.py` = the clerk who reads the manual and answers questions
- `eligibility_checker.py` = the officer who checks your application
- `recommendation_engine.py` = the advisor who tells you how to improve
- `agent.py` = the manager who decides which staff member to send your query to
- `app.py` = the front desk (what customers interact with)

---

## 📄 Step 1: The Policy Document — `data/sample_policy.txt`

This is just a plain text file containing the bank's rules written in natural language:

```
Minimum Monthly Salary: ₹30,000
Minimum Credit Score: 700
Required Documents: Aadhar card, PAN card...
```

**Why plain text?** Because the AI reads and understands natural language — no database needed. You can simply add more `.txt` or `.pdf` files here to expand the knowledge base.

---

## 🔍 Step 2: RAG Pipeline — `src/rag_pipeline.py`

**RAG = Retrieval-Augmented Generation**. Instead of the AI guessing answers, it *looks up* the policy document first and then answers. Think of it like an open-book exam vs closed-book.

### How it works step by step:

**Step 1 — Load the document**
```python
loader = TextLoader(str(file_path), encoding="utf-8")
documents.extend(loader.load())
```
Reads `sample_policy.txt` into memory as a LangChain `Document` object.

---

**Step 2 — Split into chunks**
```python
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)
```
The document is too long to send to the AI all at once. So it gets cut into smaller pieces (chunks) of ~1000 characters. The `overlap=200` means each chunk shares 200 characters with the next one — this prevents important sentences from getting cut off at boundaries.

```
[Chunk 1: chars 0-1000]
[Chunk 2: chars 800-1800]   ← overlaps with chunk 1
[Chunk 3: chars 1600-2600]  ← overlaps with chunk 2
```

---

**Step 3 — Create embeddings and store in FAISS**
```python
embeddings = FakeEmbeddings(size=1536)
vector_store = FAISS.from_documents(chunks, embeddings)
vector_store.save_local("faiss_index")
```
- **Embeddings** convert text into a list of numbers (a "vector") that captures the *meaning* of the text. Similar sentences get similar numbers.
- **FAISS** is a fast search library (by Meta) that stores these vectors and lets you find the most similar ones quickly.
- `FakeEmbeddings` generates random vectors — used here since we're not using OpenAI for embeddings. In production you'd swap this for real semantic embeddings.
- The index is **saved to disk** in `faiss_index/` so it doesn't need to rebuild every time the app starts.

---

**Step 4 — Build the QA chain**
```python
chain = RetrievalQA.from_chain_type(
    llm=llm,               # Claude Haiku does the answering
    retriever=retriever,   # FAISS finds the relevant chunks
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
)
```
When a user asks *"What is the minimum salary?"*:
1. FAISS searches for the 4 most relevant chunks from the policy doc
2. Those chunks are inserted into the prompt as `{context}`
3. Claude Haiku reads the context and generates the answer

The custom prompt instructs Claude:
> *"Answer ONLY based on the provided context. If not found, say 'I don't have that information.'"*

This prevents the AI from **hallucinating** (making up) answers.

---

## ✅ Step 3: Eligibility Checker — `src/eligibility_checker.py`

This module checks whether an applicant qualifies using **hard business rules** — no AI needed for the decision itself, only for formatting the response nicely.

### The business rules:
```python
age >= 21                          # Must be adult
salary >= 30000                    # Minimum monthly income
credit_score >= 700                # Creditworthiness
loan_amount <= salary * 10         # Can't borrow more than 10x salary
```

### How it works:

**Step 1 — Parse the input JSON**
```python
applicant = json.loads(input_json)
# input_json = '{"name":"John","age":30,"salary":60000,...}'
```
The tool receives applicant details as a JSON string.

---

**Step 2 — Apply rules and collect results**
```python
age_pass = age >= 21
rules.append({
    "rule": "Age >= 21 years",
    "passed": age_pass,
    "message": f"Age is {age} — satisfies minimum age requirement"
})
```
Each rule is checked independently and stored with a pass/fail flag and a human-readable message.

---

**Step 3 — Send results to Claude for formatting**
```python
chain = LLMChain(llm=llm, prompt=prompt)
response = chain.run(
    applicant_profile=applicant_profile,
    rule_results=rule_summary,
    verdict=verdict,
)
```
The raw rule results (PASS/FAIL) are given to Claude Haiku, which writes a professional, empathetic response. This is the difference between:

**Without AI:** `"FAIL: salary < 30000"`

**With AI:** *"Unfortunately, your monthly salary of ₹25,000 does not meet our minimum requirement of ₹30,000. We recommend..."*

---

## 💡 Step 4: Recommendation Engine — `src/recommendation_engine.py`

Only triggered when an applicant is **NOT ELIGIBLE**. It takes the failed rules and generates personalized advice.

```python
def generate_recommendations(applicant: dict, failed_rules: list[dict]) -> str:
```

The prompt tells Claude:
> *"Give a numbered list of specific, actionable recommendations. Include realistic timeframes. Suggest alternative loan amounts. End with encouragement."*

**Example input to Claude:**
```
Failed Criteria:
- Credit Score >= 700: Score is 650, below minimum of 700
- Monthly Salary >= ₹30,000: Salary is ₹25,000
```

**Example output from Claude:**
```
1. Improve your credit score above 700 (estimated 3-6 months):
   - Pay all bills on time
   - Reduce credit card utilization below 30%

2. Increase monthly income to ₹30,000+:
   - Consider a part-time income source
   - Apply for a salary increment

You're on the right track! With these improvements, reapply in 6 months.
```

---

## 🤖 Step 5: The Agent — `src/agent.py`

This is the **most important and advanced** part. The agent is like a smart router — it reads the user's message and decides which tool to use.

### The three tools registered:
```python
Tool(name="loan_policy_qa",        # For policy questions
Tool(name="eligibility_checker",   # For eligibility checks
Tool(name="recommendation_engine"  # For improvement advice
```

### How the ReAct agent thinks (Reasoning + Acting):

When a user asks: *"Is someone with salary ₹50,000 and credit score 720 eligible for a ₹4 lakh loan?"*

```
Thought: The user wants to check eligibility. I should use the eligibility_checker tool.
Action: eligibility_checker
Action Input: {"name":"Customer","age":30,"salary":50000,"loan_amount":400000,"credit_score":720}
Observation: ELIGIBLE — all criteria passed.
Thought: I have the answer.
Final Answer: Yes, this customer is eligible! All criteria are satisfied...
```

The agent **reasons step by step** before acting — this is called the **ReAct pattern**.

### Memory:
```python
memory = ConversationBufferWindowMemory(k=5)
```
The agent remembers the last 5 exchanges. So if you ask *"Why was I rejected?"* after submitting your profile, it knows who "I" refers to — it doesn't ask you to repeat yourself.

---

## 🖥️ Step 6: The Streamlit App — `app.py`

Streamlit turns Python code into a web UI with almost no web development needed.

### Key concepts used:

**`@st.cache_resource`** — Expensive operations (building the RAG chain, loading the agent) only run **once** when the app starts, not on every user interaction:
```python
@st.cache_resource
def get_rag_chain():
    return build_rag_chain()  # runs only once, cached after
```

**`st.session_state`** — Stores data between interactions (like chat history):
```python
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
```

**Two tabs:**
- **Tab 1** — Chat interface → calls `rag_chain.invoke({"query": prompt})`
- **Tab 2** — Form → collects inputs → calls `get_raw_eligibility_result()` → if not eligible, automatically calls `generate_recommendations()`

---

## 🔑 Step 7: Environment Variables — `.env`

```
ANTHROPIC_API_KEY=sk-...        ← authenticates you with the gateway
ANTHROPIC_BASE_URL=https://...  ← where to send API requests (custom gateway)
ANTHROPIC_MODEL=global.ant...   ← which model to use
```

Every source file loads these at startup:
```python
from dotenv import load_dotenv
load_dotenv()
os.getenv("ANTHROPIC_API_KEY")  # reads from .env
```

This means **no key is hardcoded** in the Python files — making the code safe to share publicly while keeping credentials private.

---

## 🔄 End-to-End Flow Summary

```
User types question
       ↓
  app.py (Streamlit UI)
       ↓
  agent.py (ReAct Agent)
       ↓
  Decides which tool to call
       ↓
  ┌────────────────────────────────────┐
  │ loan_policy_qa                     │
  │   → FAISS finds relevant chunks    │
  │   → Claude reads chunks + answers  │
  ├────────────────────────────────────┤
  │ eligibility_checker                │
  │   → Python applies rules           │
  │   → Claude formats the response    │
  ├────────────────────────────────────┤
  │ recommendation_engine              │
  │   → Claude generates advice        │
  └────────────────────────────────────┘
       ↓
  Response displayed in Streamlit
```

---

## 🧠 Key LangChain Concepts Used

| Concept | What it does | Used in |
|---|---|---|
| `TextLoader` | Reads `.txt` files into LangChain | `rag_pipeline.py` |
| `RecursiveCharacterTextSplitter` | Breaks documents into chunks | `rag_pipeline.py` |
| `FAISS` | Stores and searches text vectors | `rag_pipeline.py` |
| `RetrievalQA` | Combines retriever + LLM for Q&A | `rag_pipeline.py` |
| `PromptTemplate` | Reusable prompt with variables | All `src/` files |
| `LLMChain` | Runs a prompt through an LLM | `eligibility_checker.py` |
| `Tool` | Wraps a function for the agent to use | `agent.py` |
| `create_react_agent` | Builds a reasoning+acting agent | `agent.py` |
| `ConversationBufferWindowMemory` | Remembers last N messages | `agent.py` |
| `ChatAnthropic` | Connects to Claude via LangChain | All `src/` files |
