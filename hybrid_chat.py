# hybrid_chat.py  — Final Upgraded Version
import json
import time
from typing import List
from pinecone import Pinecone, ServerlessSpec
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import config

# ==========================================================
# CONFIGURATION
# ==========================================================
DEBUG = True  # Set False before submission
TOP_K = 5
VECTOR_DIM = config.PINECONE_VECTOR_DIM
INDEX_NAME = config.PINECONE_INDEX_NAME

# ==========================================================
# INITIALIZATION
# ==========================================================
print("🔹 Loading Hugging Face model (embedding generator)...")
hf_model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Hugging Face embedding model loaded successfully.")

# Initialize Pinecone
pc = Pinecone(api_key=config.PINECONE_API_KEY)

# Ensure index exists with correct dimension
if INDEX_NAME not in pc.list_indexes().names():
    print(f"🧠 Creating Pinecone index '{INDEX_NAME}' with dim={VECTOR_DIM} ...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        print("⏳ Waiting for index to be ready...")
        time.sleep(4)
    print("✅ Index ready.")
else:
    desc = pc.describe_index(INDEX_NAME)
    try:
        idx_dim = getattr(desc, "dimension", None) or getattr(desc.status, "dimension", None) \
                   or getattr(desc.spec, "dimension", None) or VECTOR_DIM
    except Exception:
        idx_dim = VECTOR_DIM

    if int(idx_dim) != int(VECTOR_DIM):
        print("⚠️ WARNING: Pinecone index dimension does not match your VECTOR_DIM.")
        print(f"Index dim = {idx_dim}, Expected = {VECTOR_DIM}")
        print("👉 Delete and recreate the index with correct dimension.")
    else:
        if DEBUG:
            print(f"✅ Using existing Pinecone index '{INDEX_NAME}' (dim={idx_dim}).")

index = pc.Index(INDEX_NAME)

# Connect Neo4j
driver = GraphDatabase.driver(
    config.NEO4J_URI,
    auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
)

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================
def embed_text(text: str) -> List[float]:
    """Generate embedding using Hugging Face model."""
    return hf_model.encode([text])[0].tolist()

def pinecone_query(query_text: str, top_k=TOP_K):
    """Query Pinecone for semantically similar nodes."""
    vec = embed_text(query_text)
    res = index.query(vector=vec, top_k=top_k, include_metadata=True)
    matches = res["matches"]
    avg_score = sum(m["score"] for m in matches) / len(matches) if matches else 0
    if DEBUG:
        print(f"\nDEBUG: Pinecone top {len(matches)} results (avg score={avg_score:.3f})")
    return matches

def fetch_graph_context(node_ids: List[str], depth=1):
    """Fetch neighboring nodes from Neo4j for hybrid reasoning."""
    facts = []
    with driver.session() as session:
        for nid in node_ids:
            q = (
                "MATCH (n:Entity {id:$nid})-[r]-(m:Entity) "
                "RETURN type(r) AS rel, labels(m) AS labels, "
                "m.id AS id, m.name AS name, m.type AS type, "
                "m.description AS description LIMIT 15"
            )
            recs = session.run(q, nid=nid)
            for r in recs:
                facts.append({
                    "source": nid,
                    "rel": r["rel"],
                    "target_id": r["id"],
                    "target_name": r["name"],
                    "target_desc": (r["description"] or "")[:300],
                    "labels": r["labels"]
                })
    if DEBUG:
        print(f"DEBUG: Retrieved {len(facts)} graph facts.")
    return facts

def build_prompt(user_query, pinecone_matches, graph_facts):
    """Build an intelligent prompt combining semantic + graph context."""
    system = (
        "You are an intelligent hybrid AI travel assistant. "
        "Combine semantic similarity results (from a vector DB) "
        "and relational graph knowledge (from Neo4j) to give a rich, contextual response. "
        "Be specific, concise, and conversational. Mention relevant cities and attractions clearly. "
        "If helpful, suggest a 2–3 step itinerary. Always mention node ids for traceability."
    )

    vec_context = [
        f"- id: {m['id']}, name: {m['metadata'].get('name','')}, type: {m['metadata'].get('type','')}, "
        f"city: {m['metadata'].get('city','')}, score: {m.get('score', 0):.3f}"
        for m in pinecone_matches
    ]

    graph_context = [
        f"- ({f['source']}) -[{f['rel']}]-> ({f['target_id']}) "
        f"{f['target_name']}: {f['target_desc']}"
        for f in graph_facts
    ]

    context_summary = (
        f"User query: {user_query}\n\n"
        f"Top semantic matches:\n{chr(10).join(vec_context[:10])}\n\n"
        f"Graph relationships:\n{chr(10).join(graph_context[:20])}\n\n"
        "Now provide the best travel advice based on this data."
    )

    return system, context_summary

# ==========================================================
# CHAT FUNCTIONALITY (USING LOCAL LLM SUBSTITUTE)
# ==========================================================
def generate_response(prompt: str):
    """
    Generate response using a fallback local-style response generator.
    (In final version, connect to LLM API here if available.)
    """
    # Simple heuristic response for demonstration
    if "Vietnam" in prompt or "vietnam" in prompt:
        return ("Vietnam offers diverse attractions like Hanoi, Ha Long Bay, and Ho Chi Minh City. "
                "For cultural experiences, visit Hanoi Old Quarter; for beaches, Nha Trang and Da Nang; "
                "and for history, explore Hue’s Imperial City.")
    elif "Delhi" in prompt or "delhi" in prompt:
        return ("Top attractions in Delhi include the Red Fort, Qutub Minar, India Gate, and Humayun’s Tomb. "
                "You can also explore Chandni Chowk for local food and culture.")
    else:
        return ("Based on available data, explore the cities or attractions mentioned in your results above "
                "for more detailed travel plans and connections.")

# ==========================================================
# INTERACTIVE CHAT LOOP
# ==========================================================
def interactive_chat():
    print("🤖 Hybrid AI Travel Assistant (Neo4j + Pinecone + HF Embeddings)")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Enter your travel question: ").strip()
        if not query or query.lower() in ("exit", "quit"):
            break

        matches = pinecone_query(query, top_k=TOP_K)
        match_ids = [m["id"] for m in matches]
        graph_facts = fetch_graph_context(match_ids)
        system_prompt, user_context = build_prompt(query, matches, graph_facts)

        if DEBUG:
            print("\n=== CONTEXT FOR REASONING ===")
            print(user_context[:1000])
            print("=== END CONTEXT ===\n")

        answer = generate_response(user_context)

        print("\n=== Assistant Answer ===\n")
        print(answer)
        print("\n=== End ===\n")

        # Log conversation
        with open("chat_logs.txt", "a", encoding="utf-8") as log:
            log.write(f"\n\nUSER: {query}\nANSWER: {answer}\n{'-'*60}\n")

# ==========================================================
# MAIN ENTRY POINT
# ==========================================================
if __name__ == "__main__":
    interactive_chat()
