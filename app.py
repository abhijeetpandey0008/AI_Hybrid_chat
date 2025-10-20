# app.py — Hybrid AI Travel Assistant (Safe Summarizer + Memory + Dashboard)
import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from neo4j import GraphDatabase
from transformers import pipeline
from gtts import gTTS
import tempfile, time, traceback
from streamlit_mic_recorder import mic_recorder
from pyvis.network import Network
from collections import Counter
import numpy as np
import config

# -----------------------------
# CONFIGURATION
# -----------------------------
INDEX_NAME = config.PINECONE_INDEX_NAME
NEO4J_URI = config.NEO4J_URI
NEO4J_USER = config.NEO4J_USER
NEO4J_PASSWORD = config.NEO4J_PASSWORD
PINECONE_API_KEY = config.PINECONE_API_KEY
VECTOR_DIM = config.PINECONE_VECTOR_DIM

st.set_page_config(page_title="🌏 Hybrid AI Travel Assistant", page_icon="🧭", layout="wide")

# -----------------------------
# INIT & CACHE
# -----------------------------
@st.cache_resource
def load_clients():
    hf_model = SentenceTransformer("all-MiniLM-L6-v2")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    # ✅ New, safer summarizer model trained on conversational data
    summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")
    return hf_model, pc, driver, summarizer

hf_model, pc, driver, summarizer = load_clients()

@st.cache_resource
def get_index():
    existing = pc.list_indexes().names()
    if INDEX_NAME not in existing:
        pc.create_index(name=INDEX_NAME, dimension=VECTOR_DIM, metric="cosine")
        time.sleep(5)
    return pc.Index(INDEX_NAME)

index = get_index()

# -----------------------------
# SESSION STATE
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "analytics" not in st.session_state:
    st.session_state.analytics = {
        "queries": 0,
        "pinecone_hits": [],
        "graph_facts": [],
        "response_times": [],
        "entities": []
    }

def add_to_memory(role, text):
    st.session_state.chat_history.append((role, text))
    if len(st.session_state.chat_history) > 10:
        st.session_state.chat_history = st.session_state.chat_history[-10:]

def record_analytics(pinecone_hits, graph_facts, duration, entities):
    a = st.session_state.analytics
    a["queries"] += 1
    a["pinecone_hits"].append(len(pinecone_hits))
    a["graph_facts"].append(len(graph_facts))
    a["response_times"].append(duration)
    a["entities"].extend(entities)

def get_conversation_context():
    return "\n".join([f"{r.upper()}: {t}" for r, t in st.session_state.chat_history])

# -----------------------------
# HELPERS
# -----------------------------
def embed_text(text):
    return hf_model.encode(text).tolist()

def pinecone_query(query_text, top_k=5):
    vec = embed_text(query_text)
    res = index.query(vector=vec, top_k=top_k, include_metadata=True)
    return res.get("matches", [])

def fetch_graph_context(node_ids, depth=1):
    facts = []
    with driver.session() as session:
        for nid in node_ids:
            try:
                q = (
                    f"MATCH path=(n:Entity {{id:$nid}})-[*1..{depth}]-(m:Entity) "
                    "WITH DISTINCT m, relationships(path) AS rels "
                    "RETURN m.id AS id, m.name AS name, m.type AS type, "
                    "m.description AS description, [x IN rels | type(x)] AS rel_types "
                    "LIMIT 30"
                )
                recs = session.run(q, nid=nid)
                for r in recs:
                    facts.append({
                        "source": nid,
                        "target_id": r["id"],
                        "target_name": r["name"],
                        "target_type": r["type"],
                        "target_desc": (r["description"] or "")[:250],
                        "rel_types": r["rel_types"]
                    })
            except Exception as e:
                st.error(f"Neo4j query failed: {e}")
    return facts

# ✅ SAFER SUMMARIZER FUNCTION
def reason_answer(query, pinecone_matches, graph_facts):
    """Generate contextual response using summarizer (with fallback and safety)."""
    try:
        # Collect content from vector and graph sources
        text_parts = []
        for m in pinecone_matches:
            if "metadata" in m and m["metadata"].get("description"):
                text_parts.append(m["metadata"]["description"])
        for f in graph_facts:
            text_parts.append(f["target_desc"])

        memory_context = get_conversation_context()
        context_text = (memory_context + "\n" + " ".join(text_parts)).strip()

        # Check content length safety
        if len(context_text) < 80:
            return f"Here’s a travel suggestion: {query}. You can explore attractions related to your query for more insights."

        if len(context_text) > 3000:
            context_text = context_text[:3000]  # avoid overflow

        # Summarize safely
        summary = summarizer(context_text, max_length=180, min_length=60, do_sample=False)

        if not summary or "summary_text" not in summary[0]:
            return "I gathered travel data but couldn’t generate a summary. Try refining your question."

        return summary[0]["summary_text"]

    except Exception as e:
        st.warning(f"⚠️ Summarizer issue: {e}")
        return f"I couldn’t fully summarize your query, but {query} sounds interesting! Try rephrasing or specifying a location."

def text_to_speech(text):
    try:
        tts = gTTS(text)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmp.name)
        return tmp.name
    except Exception as e:
        st.warning(f"TTS error: {e}")
        return None

def draw_graph(graph_facts):
    net = Network(height="500px", width="100%", bgcolor="#fafafa", directed=True)
    added = set()
    for f in graph_facts:
        if f["source"] not in added:
            net.add_node(f["source"], label=f["source"], color="#00aaff")
            added.add(f["source"])
        if f["target_id"] not in added:
            net.add_node(f["target_id"], label=f["target_name"], color="#ffaa00")
            added.add(f["target_id"])
        rel = ", ".join(f["rel_types"])
        net.add_edge(f["source"], f["target_id"], label=rel)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net.save_graph(tmp.name)
    with open(tmp.name, "r", encoding="utf-8") as f:
        html = f.read()
    st.components.v1.html(html, height=550, scrolling=True)

# -----------------------------
# STREAMLIT INTERFACE
# -----------------------------
tab1, tab2 = st.tabs(["💬 Chat Assistant", "📈 Summary Dashboard"])

with tab1:
    st.title("🧠 Hybrid AI Travel Assistant (Memory + Voice + Graph)")
    st.markdown("Ask travel questions naturally — the assistant remembers your context!")

    voice_query = mic_recorder(start_prompt="🎤 Speak", stop_prompt="⏹️ Stop", just_once=True)
    query_text = st.text_input("📝 Or type your question:", "")
    query = query_text or (voice_query["text"] if voice_query and "text" in voice_query else "")

    depth = st.slider("Graph depth", 1, 3, 1)
    top_k = st.slider("Top K Pinecone Results", 3, 10, 5)

    if st.button("💬 Ask Assistant"):
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            add_to_memory("user", query)
            start = time.time()
            try:
                matches = pinecone_query(query, top_k)
                ids = [m["id"] for m in matches]
                facts = fetch_graph_context(ids, depth)
                answer = reason_answer(query, matches, facts)
                add_to_memory("assistant", answer)
                duration = time.time() - start
                record_analytics(matches, facts, duration, [m["metadata"].get("name", "") for m in matches])

                st.subheader("💬 Assistant Answer")
                st.write(answer)
                audio_file = text_to_speech(answer)
                if audio_file:
                    st.audio(audio_file, format="audio/mp3")

                st.subheader("🔗 Graph Relationships")
                if len(facts) > 0:
                    draw_graph(facts)
                else:
                    st.info("No related nodes found.")
            except Exception as e:
                st.error(f"Error during processing: {e}")
                st.text(traceback.format_exc())

    if st.session_state.chat_history:
        st.subheader("🕑 Conversation History")
        for role, msg in st.session_state.chat_history:
            st.markdown(f"**{'🧍 You' if role=='user' else '🤖 Assistant'}:** {msg}")

# -----------------------------
# DASHBOARD TAB
# -----------------------------
with tab2:
    st.title("📊 Interaction Summary Dashboard")

    a = st.session_state.analytics
    if a["queries"] == 0:
        st.info("No data yet — ask a few questions first!")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Queries", a["queries"])
        col2.metric("Avg Pinecone Hits", round(np.mean(a["pinecone_hits"]), 2))
        col3.metric("Avg Graph Facts", round(np.mean(a["graph_facts"]), 2))
        col4.metric("Avg Response Time (s)", round(np.mean(a["response_times"]), 2))

        top_entities = Counter(a["entities"]).most_common(5)
        st.subheader("🏙️ Top Queried Entities")
        if top_entities:
            for name, count in top_entities:
                st.write(f"- **{name}** — {count} times")
        else:
            st.info("No entities extracted yet.")

        st.subheader("📈 Response Time Trend")
        st.line_chart(a["response_times"])
