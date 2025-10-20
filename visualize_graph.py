# visualize_graph.py
from neo4j import GraphDatabase
from pyvis.network import Network
import networkx as nx
import config

NEO_BATCH = 500  # number of relationships to fetch / visualize

driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))

def fetch_subgraph(tx, limit=500):
    # fetch nodes and relationships up to a limit
    q = (
        "MATCH (a:Entity)-[r]->(b:Entity) "
        "RETURN a.id AS a_id, labels(a) AS a_labels, a.name AS a_name, "
        "b.id AS b_id, labels(b) AS b_labels, b.name AS b_name, type(r) AS rel "
        "LIMIT $limit"
    )
    return list(tx.run(q, limit=limit))

def build_pyvis(rows, output_html="neo4j_viz.html"):
    net = Network(height="900px", width="100%", notebook=False, directed=True)
    for rec in rows:
        a_id = rec["a_id"]; a_name = rec["a_name"] or a_id
        b_id = rec["b_id"]; b_name = rec["b_name"] or b_id
        a_labels = rec["a_labels"]; b_labels = rec["b_labels"]
        rel = rec["rel"]

        net.add_node(a_id, label=f"{a_name}\n({','.join(a_labels)})", title=f"{a_name}")
        net.add_node(b_id, label=f"{b_name}\n({','.join(b_labels)})", title=f"{b_name}")
        net.add_edge(a_id, b_id, title=rel)

    net.show(output_html, notebook=False)
    print(f"Saved visualization to {output_html}")

def main():
    with driver.session() as session:
        rows = session.execute_read(fetch_subgraph, limit=NEO_BATCH)
    build_pyvis(rows)

if __name__ == "__main__":
    main()
