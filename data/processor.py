import re
from .db import get_connection

# Configuration

# direct lookup table. arXiv category 
CATEGORY_MAP = {
    "cs.CV" : "Computer Vision",
    "cs.CL" : "NLP",
    "cs.LG" : "Machine Learning",
    "cs.AI" : "Artificial Intelligence",
    "cs.NE" : "Neural Networks",
    "cs.RO" : "Robotics",
    "cs.IR" : "Information Retrieval",
    "cs.CR" : "Cybersecurity",
    "eess.IV": "Medical Imaging",
    "q-bio" : "Bioinformatics",
    "stat.ML": "Machine Learning",
}

#  fallback for when categories are ambiguous or too broad
KEYWORD_MAP = {
    "Computer Vision"   : ["image classification", "object detection", "segmentation", "yolo", "vision transformer"],
    "NLP"               : ["natural language", "language model", "text generation", "bert", "gpt", "sentiment"],
    "Reinforcement Learning": ["reinforcement learning", "reward function", "policy gradient", "q-learning", "markov"],
    "Graph Neural Networks" : ["graph neural", "gnn", "node embedding", "message passing", "graph convolution"],
    "Generative AI"     : ["generative", "diffusion model", "gan", "stable diffusion", "image generation", "vae"],
    "Healthcare"        : ["medical", "clinical", "healthcare", "disease", "diagnosis", "patient", "drug"],
    "Cybersecurity"     : ["intrusion detection", "malware", "anomaly detection", "network security", "attack"],
    "Finance"           : ["stock", "financial", "trading", "market prediction", "portfolio"],
    "Robotics"          : ["robot", "autonomous", "navigation", "control system", "manipulation"],
    "Bioinformatics"    : ["protein", "genomic", "dna", "rna", "molecular", "biological sequence"],
}

#  disambiguation rules. The word "sequence" appears in both NLP and bioinformatics papers. If we detect NLP keywords but 
#  also see "protein" or "dna", it's probably bioinformatics, not NLP
EXCLUDE_MAP = {
    "NLP"               : ["protein", "dna", "rna"],
    "Graph Neural Networks" : ["plot", "figure", "chart"],
    "Computer Vision"   : ["audio", "speech", "acoustic"],
}

# Helpers
def classify_paper(categories: str, abstract: str) -> list[str]:
    subfields = set()
    abstract_lower = abstract.lower()

    for cat_code, subfield in CATEGORY_MAP.items():
        if cat_code in categories:
            subfields.add(subfield)

    for subfield, keywords in KEYWORD_MAP.items():
        if subfield in subfields:
            continue

        has_keyword = any(kw in abstract_lower for kw in keywords)
        if not has_keyword:
            continue

        exclusions = EXCLUDE_MAP.get(subfield, [])
        is_excluded = any(ex in abstract_lower for ex in exclusions)
        if not is_excluded:
            subfields.add(subfield)

    return sorted(subfields) if subfields else ["General Machine Learning"]

#  we use a set, not a list, because a paper might match cs.LG in CATEGORY_MAP AND "machine learning" in KEYWORD_MAP.
#  A set automatically prevents duplicates.   

# any(kw in abstract_lower for kw in keywords) — generator expression that short-circuits. As soon as one keyword matches, it stops checking


# Classification Runner

def run_classification():
    conn   = get_connection()
    cursor = conn.cursor()

    papers = cursor.execute(
        "SELECT id, categories, abstract FROM papers WHERE subfields IS NULL"
    ).fetchall()
    # WHERE subfields IS NULL — only processes papers that haven't been classified yet. Safe to re-run without double-processing. 
    # Same principle as INSERT OR IGNORE in the scraper

    print(f"Classifying {len(papers)} papers...")
    updated = 0

    for paper in papers:
        subfields = classify_paper(paper["categories"], paper["abstract"])
        subfields_str = ",".join(subfields)
        # ",".join(subfields) — converts ["Computer Vision", "NLP"] to "Computer Vision,NLP" for storage. 
        # SQLite has no array type, so comma-separated strings are the clean solution.

        cursor.execute(
            "UPDATE papers SET subfields = ? WHERE id = ?",
            (subfields_str, paper["id"])
        )
        updated += 1
        # UPDATE papers SET subfields = ? WHERE id = ? — ? is a positional placeholder. SQLite replaces them with the tuple values 
        # (subfields_str, paper["id"]) in order

    conn.commit()
    # Commit once after all updates — not inside the loop. 
    # Committing inside the loop would write to disk 300 times. One commit at the end is far faster
    conn.close()
    print(f"Classification complete. Updated {updated} papers.")


if __name__ == "__main__":
    run_classification()