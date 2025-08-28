GraphMind
<img src="logo.png" alt="GraphMind logo" width="160"/>

GraphMind is a Streamlit app that turns a PDF into a Neo4j knowledge graph and lets you chat over it using a Groq-hosted LLM. It also includes a vector fallback so questions can be answered from the PDF text if the graph route misses.

Features

📄 PDF → Graph: Split pages, extract entities/relations (LangChain LLMGraphTransformer), write into Neo4j.

🧠 Cypher QA: LLM generates Cypher via GraphCypherQAChain against your graph schema.

🧭 Vector fallback: Optional hybrid search with Neo4jVector + HuggingFaceEmbeddings.

💬 Chat UI: Lightweight Streamlit chat with user/bot avatars.

⚙️ Manual connect: Groq API + Neo4j credentials are prefilled from .env but only connect when you click.

Tech stack

Frontend: Streamlit

LLM: Groq (llama-3.3-70b-versatile)

Embeddings: sentence-transformers (all-MiniLM-L6-v2 via langchain-huggingface)

Graph DB: Neo4j 5.x

Orchestration: LangChain (graph_experimental, graph_qa.cypher, vectorstores.Neo4jVector)

Quick start
1) Prerequisites

Python 3.10+ (3.11 recommended)

A running Neo4j 5 instance (local or Aura) with URL, username, password

A Groq account and API key

2) Clone & set up

Windows (PowerShell):

# from your desired parent folder
git clone https://github.com/<YOUR_USERNAME>/GraphMind.git
cd GraphMind

py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt


macOS/Linux:

git clone https://github.com/<YOUR_USERNAME>/GraphMind.git
cd GraphMind

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

3) Configure environment

Copy the template and fill in values:

cp .env.example .env


Set the variables:

GROQ_API_KEY=...
NEO4J_URL=bolt+s://<host>:<port>   # or bolt:// for local
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=...

4) Run
streamlit run app.py

How to use (in-app flow)

Screen 1 – Connect

Enter Groq API key → Connect Groq API

Enter Neo4j URL / Username / Password → Connect to Neo4j

When both are connected, the app switches to Screen 2.

Screen 2 – Ingest & Chat

Upload a PDF. The app:

Loads & splits text (200 chars, 40 overlap)

Converts chunks into graph documents (LLMGraphTransformer)

Writes nodes/edges into Neo4j

(Best-effort) configures Neo4jVector for hybrid search

Ask questions in the chat:

First try: Cypher-generated query over the graph

Fallback: Vector similarity over PDF chunks

Project structure
GraphMind/
├─ app.py                # Streamlit app (UI, ingest pipeline, QA)
├─ requirements.txt
├─ .env.example          # Template for secrets (copy to .env)
├─ .gitignore
├─ logo.png              # Logo shown in header
├─ bot-avatar.svg        # Chat bot avatar
└─ user-avatar.svg       # Chat user avatar

Configuration notes / safety

Destructive import (IMPORTANT): In the current code, PDF ingest deletes all nodes in the connected Neo4j DB before writing new ones:

MATCH (n) DETACH DELETE n;


Use a dedicated database for this app, or remove/replace that line if you need to preserve data.

Cypher write protection: The QA chain is created with allow_dangerous_requests=True. For safety, set it to False and/or validate generated Cypher to only allow MATCH/RETURN/... (no CREATE/DELETE/SET).

Avatars & logo: If images are missing locally, update the paths in app.py or add fallbacks.

Troubleshooting

Cannot connect to Neo4j: Verify URL protocol (bolt:// for local, bolt+s:// for Aura), credentials, and that the DB is running.

Hybrid search disabled: If index initialization fails, the app falls back to graph-only QA; check Neo4j version/permissions.

“Unknown” answers: The app will try the vector fallback; if still empty, the data may not include what you’re asking—try a broader question.

Roadmap / ideas (optional)

Non-destructive imports (batch IDs and “clear current batch” button)

Turn off dangerous Cypher by default + query sanitizer

Explicit index creation & readiness checks for Neo4jVector

Move assets to assets/ and reference paths accordingly

Add tests/CI (ruff/pytest) and a LICENSE (MIT)

License

Add a license you prefer (MIT is common for personal projects).

Acknowledgements

Built with LangChain, Neo4j, Streamlit, Groq, and sentence-transformers.
