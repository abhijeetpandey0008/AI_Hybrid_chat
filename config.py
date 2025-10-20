# config_example.py — copy to config.py and fill with real values.
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "BlueEnigmaDB"

OPENAI_API_KEY = "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" # your OpenAI API key

PINECONE_API_KEY = "pcsk_5ZkgSx_BjuKk4xrtzsKVzwzKm8ZLShXHo1mWkNsGuMJyx9HeQnVvmFKYrPn5TDXXmn2Hdu" # your Pinecone API key
PINECONE_ENV = "us-east-1"   # example
PINECONE_INDEX_NAME = "vietnam-travel"
PINECONE_VECTOR_DIM = 384       # adjust to embedding model used (text-embedding-3-large ~ 3072? check your model); we assume 1536 for common OpenAI models — change if needed.
