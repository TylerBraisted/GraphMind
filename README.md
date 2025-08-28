# GraphMind

<img src="logo.png" alt="GraphMind logo" width="160"/>

GraphMind is a Streamlit app that turns a PDF into a **Neo4j knowledge graph** and lets you **chat** over it using a Groq-hosted LLM. It also includes a **vector fallback** so questions can be answered from the PDF text if the graph route misses.

---

## Features

- 📄 **PDF → Graph**: Split pages, extract entities/relations (LangChain `LLMGraphTransformer`), write into Neo4j.
- 🧠 **Cypher QA**: LLM generates Cypher via `GraphCypherQAChain` against your graph schema.
- 🧭 **Vector fallback**: Optional hybrid search with `Neo4jVector` + `HuggingFaceEmbeddings`.
- 💬 **Chat UI**: Lightweight Streamlit chat with user/bot avatars.
- ⚙️ **Manual connect**: Groq API + Neo4j credentials are **prefilled from `.env`** but only connect when you click.

---

## Tech stack

- **Frontend:** Streamlit  
- **LLM:** Groq (`llama-3.3-70b-versatile`)  
- **Embeddings:** `sentence-transformers` (all-MiniLM-L6-v2 via `langchain-huggingface`)  
- **Graph DB:** Neo4j 5.x  
- **Orchestration:** LangChain (`graph_experimental`, `graph_qa.cypher`, `vectorstores.Neo4jVector`)

---

## Quick start

### 1) Prerequisites
- Python **3.10+** (3.11 recommended)
- A running **Neo4j 5** instance (local or Aura) with URL, username, password
- A **Groq** account and API key

### 2) Clone & set up

**Windows (PowerShell):**
```powershell
# from your desired parent folder
git clone https://github.com/<YOUR_USERNAME>/GraphMind.git
cd GraphMind

py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS/Linux:**
```bash
git clone https://github.com/<YOUR_USERNAME>/GraphMind.git
cd GraphMind

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
### 3) Configure enviornment
Copy the template and fill in values:

cp .env.example .env


Set the variables in .env:

GROQ_API_KEY=...
NEO4J_URL=bolt+s://<host>:<port>   # or bolt:// for local
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=...

### 4) Run
streamlit run app.py
