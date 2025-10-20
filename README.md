# ✨ Hybrid AI Travel Assistant  
> 🧠 *An intelligent hybrid system combining Graph + Vector + LLM for contextual travel recommendations*

---

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python Version"/>
  <img src="https://img.shields.io/badge/Neo4j-GraphDB-008CC1?logo=neo4j&logoColor=white" alt="Neo4j"/>
  <img src="https://img.shields.io/badge/HuggingFace-MiniLM-yellow?logo=huggingface&logoColor=black" alt="HuggingFace"/>
  <img src="https://img.shields.io/badge/Pinecone-VectorDB-6B4EFF?logo=pinecone&logoColor=white" alt="Pinecone"/>
  <img src="https://img.shields.io/badge/License-MIT-green?logo=open-source-initiative&logoColor=white" alt="License"/>
</p>

---

## 🌍 Overview
This project builds a **Hybrid Knowledge Reasoning System** for travel assistance — combining the power of **Graph Databases**, **Vector Search**, and **Language Models**.

### 🧩 Core Components
- 🕸️ **Neo4j** → Understands structured relationships (cities ↔ attractions)  
- 🧭 **Pinecone** → Retrieves semantically similar concepts  
- 🤗 **Hugging Face (MiniLM)** → Generates local text embeddings (offline & free)  
- 💬 **LLM (Mistral/GPT)** → Produces fluent, context-aware travel answers  

Example user query:
> “Suggest the best tourist destinations in Vietnam with nearby cultural spots.”

---

## 🧱 Architecture

```
                ┌────────────────────────────┐
                │         USER QUERY          │
                └────────────┬───────────────┘
                             │
                             ▼
           ┌─────────────────────────────────────┐
           │  🧠 SentenceTransformer (MiniLM-L6)  │
           │   → Text Embeddings (384-dim)       │
           └─────────────────────────────────────┘
                             │
                             ▼
                ┌────────────────────────────┐
                │    🧭 Pinecone Vector DB     │
                │  → Retrieves semantic nodes │
                └────────────────────────────┘
                             │
                             ▼
                ┌────────────────────────────┐
                │     🕸️ Neo4j Graph DB       │
                │  → Expands contextual data  │
                └────────────────────────────┘
                             │
                             ▼
                ┌────────────────────────────┐
                │ 🤖 LLM Response Generator   │
                │  → Summarizes & recommends  │
                └────────────────────────────┘
```

---

## ⚙️ Setup Instructions

### 🪄 1. Clone the Repository
```bash
git clone https://github.com/abhijeetpandey-dev/Hybrid-AI-Travel-Assistant.git
cd Hybrid-AI-Travel-Assistant
```

### 📦 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**Core Libraries**
```
neo4j
pinecone-client
sentence-transformers
torch
tqdm
```

*(Optional: for OpenAI usage)*  
```
openai
```

---

### 🔐 3. Configuration
Create a file named **`config.py`** in the project root:
```python
PINECONE_API_KEY = "your_pinecone_key"
PINECONE_INDEX_NAME = "vietnam-travel"
PINECONE_VECTOR_DIM = 384  # for MiniLM model

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_password"

# Optional (if using OpenAI for chat)
OPENAI_API_KEY = "your_openai_api_key"
```

---

## 🧩 Dataset Preparation

### 🗂️ `vietnam_travel_dataset.json`
This dataset holds cities, attractions, and regional info:
```json
{
  "id": "city_hanoi",
  "type": "City",
  "name": "Hanoi",
  "region": "Northern Vietnam",
  "description": "Capital city known for lakes, temples, and culture."
}
```

---

## 🚀 Running the Project

### 🔹 Step 1: Upload to Pinecone
```bash
python pinecone_upload.py
```
✅ Generates 384-dim local embeddings  
✅ Uploads batches to Pinecone  

---

### 🔹 Step 2: Build Neo4j Graph
In the Neo4j Browser or script:
```cypher
CREATE (hanoi:City {id:"city_hanoi", name:"Hanoi"});
CREATE (halong:Attraction {id:"attr_halong", name:"Ha Long Bay"});
CREATE (hanoi)-[:CONNECTED_TO]->(halong);
```

---

### 🔹 Step 3: Run the Hybrid Chat
```bash
python hybrid_chat.py
```

Example:
```
Enter your travel question: best places to visit in Vietnam
```

Output:
```
Top results: Hanoi, Ho Chi Minh City, Da Nang

✨ Suggested itinerary:
1️⃣ Start in Hanoi for historical culture
2️⃣ Visit Da Nang for coastal experiences
3️⃣ End in Ho Chi Minh City for nightlife and street food
```

---

## 🧠 How It Works

| Layer | Function | Technology |
|-------|-----------|------------|
| 🧩 **Hugging Face MiniLM** | Convert text → 384-dim vector | SentenceTransformers |
| 🧭 **Pinecone** | Retrieve semantically similar entities | Vector Search |
| 🕸️ **Neo4j** | Graph expansion & relations | Graph Database |
| 💬 **LLM (Mistral / GPT)** | Generate coherent answers | Language Model |

---

## 🎥 Demo Video
🎬 [Video Explaination](https://drive.google.com/file/d/13yxs3Rs4jeXXK-rKHl8H85FmckCxCCCs/view?usp=sharing)  

**Showcase:**
- Dataset preparation  
- Pinecone + Neo4j connection  
- Query results  
- System explanation  

---

## 👨‍💻 Author
**Abhijeet Pandey**   
📧 [abhijeetpandey1219@gmail.com]
🌐 [GitHub: abhijeetpandey-dev](https://github.com/abhijeetpandey0008)

---

## 🪪 License
```
MIT License © 2025 Abhijeet Pandey
```

---

## 🚀 Future Enhancements
- 🌐 Integrate live APIs (TripAdvisor, Google Places)
- 🗣️ Add speech input/output for real-time travel chat
- 🧳 Expand to multi-country travel graph
- 💻 Streamlit or Gradio UI for interactive demo

---

<p align="center">
  ⭐ <b>If you found this project inspiring, consider giving it a star!</b> ⭐  
</p>
