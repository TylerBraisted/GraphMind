import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Neo4jVector
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.chains.graph_qa.cypher import GraphCypherQAChain
from langchain.chains import LLMChain
import streamlit as st
import tempfile

USER_AVATAR = "./test.svg"
BOT_AVATAR  = "./bot-avatar.svg"

# -------------------------
# Session / Styles
# -------------------------
def init_session():
    if 'screen' not in st.session_state:
        st.session_state['screen'] = 1  # 1 = Connect, 2 = App
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []
    if 'pdf_processed' not in st.session_state:
        st.session_state['pdf_processed'] = False


def show_login_styles():
    st.markdown(
        """
        <style>
            .section-title {
                font-size: 1.4rem; font-weight: 700;
                margin-bottom: 0.5rem; color: #1f2937; text-align: center;
            }
            .section-subtitle {
                color: #6b7280; font-size: 0.95rem;
                text-align: center; margin-bottom: 1.2rem;
            }
            @media (max-width: 768px) {
                [data-testid="stAppViewContainer"] .main [data-testid="stHorizontalBlock"]:first-of-type {
                    min-height: auto;
                }
                [data-testid="stAppViewContainer"] .main [data-testid="stImage"]:first-of-type img {
                    max-height: 18vh !important;
                }
            }
        </style>
        """,
        unsafe_allow_html=True
    )


def show_app_styles():
    st.markdown("""
    <style>
      .chip {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 4px 10px; border-radius: 9999px;
        background: #eef2ff; border: 1px solid #c7d2fe;
        font-weight: 600; font-size: 0.9rem;
      }
      .row { display:flex; justify-content:flex-end; gap:.5rem; }
      @media (max-width: 768px) { .row { justify-content:flex-start; } }
    </style>
    """, unsafe_allow_html=True)


def render_header():
    with st.container():
        cols = st.columns([1, 5])
        with cols[0]:
            st.image("logo.png", use_container_width=True)
        with cols[1]:
            groq = "✅" if st.session_state.get("groq_connected") else "⛔"
            neo = "✅" if st.session_state.get("neo4j_connected") else "⛔"
            st.markdown(
                f'<div class="row" style="justify-content:flex-end;">'
                f'<span class="chip">{groq}&nbsp;Groq</span>'
                f'<span class="chip">{neo}&nbsp;Neo4j</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


# -------------------------
# Helpers
# -------------------------
def compact_schema(schema_text: str, max_chars: int = 8000) -> str:
    if not schema_text:
        return schema_text
    if len(schema_text) <= max_chars:
        return schema_text
    return schema_text[:max_chars] + "\n/* …schema truncated for token budget… */"


def cypher_prompt_generic():
    template = """
You are a Cypher expert. Generate a Cypher query for Neo4j given the user question
and the database schema below.

Rules:
- Use only labels, relationship types, and properties that appear in the provided schema.
- Prefer simple, valid Cypher with MATCH ... RETURN ...
- If the user asks for the *name/title* of a node, return:
  coalesce(n.name, n.full_name, n.title, n.label, n.id, n.identifier) AS name
- When label synonyms are implied, try common alternates:
  Medication↔Drug↔Prescription; Disease↔Condition↔Illness; Doctor↔Physician; Patient↔Person.
- If a specific property is not in the schema, return properties(n) and a compact identifier as above.
- Always include a sensible LIMIT (e.g., LIMIT 100) unless the user specifies otherwise.
- Do not include explanations—output the Cypher only.

Schema:
{schema}

Question:
{question}
"""
    return PromptTemplate(template=template, input_variables=["schema", "question"])


def normalize_graph_after_ingest(graph: Neo4jGraph):
    # Ensure a human-readable name property exists
    graph.query("""
    MATCH (n)
    WITH n, coalesce(n.name, n.full_name, n.title, n.label, n.id, n.identifier) AS nm
    WHERE nm IS NOT NULL
    SET n.name = nm
    """)
    graph.query("""
    MATCH (d:Drug) WHERE NOT d:Medication SET d:Medication
    """)


# -------------------------
# App
# -------------------------
def main():
    st.set_page_config(layout="wide", page_title="GraphMind", page_icon=":graph:")
    init_session()
    load_dotenv()  # load .env into os.environ

    env_groq_key = os.getenv('GROQ_API_KEY', '')
    env_neo4j_url = os.getenv('NEO4J_URL', '')
    env_neo4j_username = os.getenv('NEO4J_USERNAME', '')
    env_neo4j_password = os.getenv('NEO4J_PASSWORD', '')

    embeddings = st.session_state.get('embeddings')
    llm = st.session_state.get('llm')
    graph = st.session_state.get('graph')

    # ---------- SCREEN 1 ----------
    if st.session_state['screen'] == 1:
        show_login_styles()
        show_app_styles()
        render_header()

        col1, col2 = st.columns(2)

        # Left: Groq API (prefilled from env, connect ONLY on button click)
        with col1:
            st.markdown('<div class="section-title">Groq API</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-subtitle">Connect your Groq API key to enable AI processing</div>', unsafe_allow_html=True)

            groq_api_key = st.text_input(
                "Groq API Key",
                value=st.session_state.get("groq_key", env_groq_key),
                type='password',
                key="groq_key"
            )
            groq_connect_button = st.button("Connect Groq API", key="groq_connect")

            if groq_connect_button:
                if not groq_api_key:
                    st.error("Please enter a Groq API key.")
                else:
                    try:
                        os.environ['GROQ_API_KEY'] = groq_api_key
                        st.session_state['GROQ_API_KEY'] = groq_api_key

                        # Initialize ONLY on click
                        st.session_state['embeddings'] = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                        st.session_state['llm'] = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, max_tokens=800)

                        embeddings = st.session_state['embeddings']
                        llm = st.session_state['llm']
                        st.session_state['groq_connected'] = True
                        st.success("Groq API Key connected successfully.")
                    except Exception as e:
                        st.error(f"Failed to connect to Groq API: {e}")


        # Right: Neo4j (prefilled from env, connect ONLY on button click)
        with col2:
            st.markdown('<div class="section-title">Neo4j Database</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-subtitle">Connect to your Neo4j graph database instance</div>', unsafe_allow_html=True)

            neo4j_url = st.text_input(
                "Neo4j URL",
                value=st.session_state.get('neo4j_url', env_neo4j_url),
                key="neo4j_url"
            )
            neo4j_username = st.text_input(
                "Neo4j Username",
                value=st.session_state.get('neo4j_username', env_neo4j_username),
                key="neo4j_username"
            )
            neo4j_password = st.text_input(
                "Neo4j Password",
                value=st.session_state.get('neo4j_password', env_neo4j_password),
                type='password',
                key="neo4j_password"
            )

            neo4j_connect_button = st.button("Connect to Neo4j", key="neo4j_connect")

            if neo4j_connect_button:
                if not neo4j_url or not neo4j_username or not neo4j_password:
                    st.error("Please fill in Neo4j URL, Username, and Password.")
                else:
                    try:
                        graph = Neo4jGraph(url=neo4j_url, username=neo4j_username, password=neo4j_password)
                        st.session_state['graph'] = graph
                        st.session_state['neo4j_connected'] = True
                        st.session_state['neo4j_url'] = neo4j_url
                        st.session_state['neo4j_username'] = neo4j_username
                        st.session_state['neo4j_password'] = neo4j_password
                        st.success("Connected to Neo4j database.")
                    except Exception as e:
                        st.error(f"Failed to connect to Neo4j: {e}")

            if st.session_state.get('neo4j_connected'):
                st.info("✅ Neo4j is connected")

        # Only advance if BOTH are connected (still no auto-connect)
        if (
            st.session_state.get('neo4j_connected') and
            st.session_state.get('groq_connected') and
            st.session_state.get('graph') is not None and
            st.session_state.get('embeddings') is not None and
            st.session_state.get('llm') is not None
        ):
            st.session_state['screen'] = 2
            st.rerun()

    # ---------- SCREEN 2 ----------
    if st.session_state['screen'] == 2:
        show_app_styles()
        render_header()

        if graph is None:
            st.warning("Please connect to the Neo4j database on Screen 1.")
            if st.button("Go back to Screen 1"):
                st.session_state['screen'] = 1
                st.rerun()
            return
        if embeddings is None or llm is None:
            st.warning("Please enter your Groq API key on Screen 1.")
            if st.button("Go back to Screen 1"):
                st.session_state['screen'] = 1
                st.rerun()
            return

        # ---- Auto Mode only: no schema mode UI ----
        if not st.session_state['pdf_processed'] and 'qa' not in st.session_state:
            uploaded_file = st.file_uploader("", type="pdf")
            if uploaded_file is not None:
                with st.spinner("Processing the PDF..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_file_path = tmp_file.name

                    loader = PyPDFLoader(tmp_file_path)
                    pages = loader.load_and_split()

                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
                    docs = text_splitter.split_documents(pages)

                    lc_docs = []
                    for doc in docs:
                        lc_docs.append(
                            Document(
                                page_content=doc.page_content.replace("\n", ""),
                                metadata={'source': uploaded_file.name, 'page': doc.metadata.get("page", None)}
                            )
                        )

                    # Clear graph
                    graph.query("MATCH (n) DETACH DELETE n;")

                    transformer = LLMGraphTransformer(
                        llm=llm,
                        node_properties=True,
                        relationship_properties=True
                    )

                    graph_documents = transformer.convert_to_graph_documents(lc_docs)
                    graph.add_graph_documents(graph_documents, include_source=True)

                    # Normalize labels/properties for better querying
                    normalize_graph_after_ingest(graph)

                    # Vector index (best-effort) + capture for fallback
                    try:
                        vstore = Neo4jVector.from_existing_graph(
                            embedding=embeddings,
                            url=st.session_state['neo4j_url'],
                            username=st.session_state['neo4j_username'],
                            password=st.session_state['neo4j_password'],
                            database="neo4j",
                            node_label="Document",
                            text_node_properties=["text"],
                            embedding_node_property="embedding",
                            index_name="vector_index",
                            keyword_index_name="entity_index",
                            search_type="hybrid"
                        )
                        st.session_state['vstore'] = vstore
                    except Exception as e:
                        st.session_state['vstore'] = None

                    # Safely patch get_schema to compact it
                    try:
                        _orig_get_schema = graph.get_schema
                        def _patched_get_schema():
                            try:
                                return compact_schema(_orig_get_schema())
                            except TypeError:
                                return compact_schema(graph.get_schema)
                        graph.get_schema = _patched_get_schema
                    except Exception:
                        pass

                    question_prompt = cypher_prompt_generic()

                    qa = GraphCypherQAChain.from_llm(
                        llm=llm,
                        graph=graph,
                        cypher_prompt=question_prompt,
                        verbose=True,
                        allow_dangerous_requests=True
                    )
                    st.session_state['qa'] = qa
                    st.session_state['pdf_processed'] = True
                    st.success("PDF uploaded successfully.")


        # Chat UI after processing
        if st.session_state.get('pdf_processed') and 'qa' in st.session_state:
            for msg in st.session_state['messages']:
                role = msg["role"]
                avatar = BOT_AVATAR if role == "assistant" else USER_AVATAR
                with st.chat_message(role, avatar=avatar):
                    st.markdown(msg["content"])

            prompt = st.chat_input("Ask a question about your graph...")
            if prompt:
                st.session_state['messages'].append({"role": "user", "content": prompt})
                with st.chat_message("user", avatar=USER_AVATAR):
                    st.markdown(prompt)

                with st.chat_message("assistant", avatar=BOT_AVATAR):
                    with st.spinner("Generating answer..."):
                        res = st.session_state['qa'].invoke({"query": prompt})
                        answer = (res.get('result') or "").strip()

                        # Fallback to vector search if graph QA doesn't know
                        if not answer or answer.lower().startswith("i don't know"):
                            vs = st.session_state.get('vstore')
                            if vs is not None:
                                try:
                                    hits = vs.similarity_search(prompt, k=4)
                                except Exception:
                                    hits = []
                                if hits:
                                    ctx = "\n\n".join([h.page_content for h in hits])
                                    fallback_prompt = PromptTemplate.from_template(
                                        "Answer the question using only the context.\n\nQuestion: {q}\n\nContext:\n{c}\n\nAnswer:"
                                    )
                                    try:
                                        answer = LLMChain(llm=llm, prompt=fallback_prompt).run({"q": prompt, "c": ctx}).strip()
                                    except Exception as e:
                                        answer = f"(Fallback failed) {e}"
                                if not answer:
                                    answer = "I couldn’t find that in the graph or the document."

                        st.markdown(answer)

                st.session_state['messages'].append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
