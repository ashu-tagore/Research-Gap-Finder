import re
from .db import get_connection

# CONFIGURATION

#  direct lookup table. arXiv category
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

# fallback for when categories are ambiguous or too broad
KEYWORD_MAP = {
    "Computer Vision"       : ["image classification", "object detection", "segmentation", "yolo", "vision transformer"],
    "NLP"                   : ["natural language", "language model", "text generation", "bert", "gpt", "sentiment"],
    "Reinforcement Learning": ["reinforcement learning", "reward function", "policy gradient", "q-learning", "markov"],
    "Graph Neural Networks" : ["graph neural", "gnn", "node embedding", "message passing", "graph convolution"],
    "Generative AI"         : ["generative", "diffusion model", "gan", "stable diffusion", "image generation", "vae"],
    "Healthcare"            : ["medical", "clinical", "healthcare", "disease", "diagnosis", "patient", "drug"],
    "Cybersecurity"         : ["intrusion detection", "malware", "anomaly detection", "network security", "attack"],
    "Finance"               : ["stock", "financial", "trading", "market prediction", "portfolio"],
    "Robotics"              : ["robot", "autonomous", "navigation", "control system", "manipulation"],
    "Bioinformatics"        : ["protein", "genomic", "dna", "rna", "molecular", "biological sequence"],
}

# disambiguation rules. The word "sequence" appears in both NLP and bioinformatics papers. 
# If we detect NLP keywords but also see "protein" or "dna", it's probably bioinformatics, not NLP.
EXCLUDE_MAP = {
    "NLP"                   : ["protein", "dna", "rna"],
    "Graph Neural Networks" : ["plot", "figure", "chart"],
    "Computer Vision"       : ["audio", "speech", "acoustic"],
}

# The specific terms we track over time. These are chosen because they represent either established techniques or emerging ones
TREND_KEYWORDS = [
    "transformer", "diffusion model", "large language model",
    "reinforcement learning", "graph neural network", "contrastive learning",
    "federated learning", "knowledge distillation", "self-supervised",
    "zero-shot", "few-shot", "prompt engineering", "fine-tuning",
    "multimodal", "vision transformer", "generative model",
    "anomaly detection", "object detection", "image segmentation",
    "drug discovery", "protein folding", "autonomous driving",
]


# CLASSIFICATION HELPERS

def classify_paper(categories: str, abstract: str) -> list[str]:
    subfields     = set() 
    # we use a set, not a list, because a paper might match cs.LG in CATEGORY_MAP AND "machine learning" in KEYWORD_MAP. 
    # A set automatically prevents duplicates

    abstract_lower = abstract.lower()

    for cat_code, subfield in CATEGORY_MAP.items():
        if cat_code in categories:
            # categories is a space-separated string like "cs.LG cs.CV stat.ML". 
            # Python's in operator checks if the substring exists in the string
            subfields.add(subfield)

    for subfield, keywords in KEYWORD_MAP.items():
        if subfield in subfields:
            #  if category mapping already assigned this subfield, skip keyword matching for it
            continue

        has_keyword = any(kw in abstract_lower for kw in keywords)
        if not has_keyword:
            continue

        exclusions  = EXCLUDE_MAP.get(subfield, []) # safely gets exclusions for this subfield. If no exclusions exist, returns empty list so the loop runs harmlessly
        is_excluded = any(ex in abstract_lower for ex in exclusions) # generator expression that short-circuits. As soon as one keyword matches, it stops checking
        if not is_excluded:
            subfields.add(subfield)

    return sorted(subfields) if subfields else ["General Machine Learning"] # returns a consistent ordering


# TREND HELPERS

def split_papers_by_period(papers: list) -> tuple[list, list]:
    sorted_papers = sorted(papers, key=lambda p: p["published_date"])
    # sorted(..., key=lambda p: p["published_date"]) — sorts papers chronologically. The lambda is an anonymous function 
    # that extracts published_date from each paper dict as the sort key. 
    # Date strings in YYYY-MM-DD format sort correctly as strings because the most significant unit (year) comes first

    midpoint      = len(sorted_papers) // 2
    return sorted_papers[:midpoint], sorted_papers[midpoint:] # Returns two lists: older half and newer half. We call them historical and recent throughout


def compute_frequency(papers: list, keyword: str) -> float:
    if not papers:
        return 0.0
    count = sum(1 for p in papers if keyword in p["abstract"].lower()) # counts how many papers contain the keyword
    return count / len(papers) # normalizes by corpus size. Result is a float between 0 and 1. Example: 30 matches in 150 papers → 0.2, meaning 20% of papers mention this keyword


def compute_growth_rate(historical_freq: float, recent_freq: float) -> float:
    if historical_freq == 0:
        return 1.0 if recent_freq > 0 else 0.0 # we can't divide by zero. If a keyword never appeared historically but appears now, that's a 100% new emergence — we return 1.0.  If it appears in neither period, growth is 0.0
    return (recent_freq - historical_freq) / historical_freq


def classify_trend_status(growth_rate: float, recent_freq: float) -> str:
    if recent_freq < 0.01: #  if fewer than 1% of recent papers mention this keyword, it's essentially gone regardless of growth rate
        return "obsolete"
    if growth_rate > 0.5: #  grew more than 50%. The gap between supply and demand is widening — more researchers interested, fewer papers covering it fully
        return "widening"
    if growth_rate < -0.2: # The gap is closing — research is catching up
        return "closing"
    return "stable"


# RUNNERS

def run_classification():
    conn   = get_connection()
    cursor = conn.cursor()

    papers  = cursor.execute(
        "SELECT id, categories, abstract FROM papers WHERE subfields IS NULL"
    ).fetchall()
    # only processes papers that haven't been classified yet. Safe to re-run without double-processing. 
    # Same principle as INSERT OR IGNORE in the scraper

    print(f"Classifying {len(papers)} papers...")
    updated = 0

    for paper in papers:
        subfields     = classify_paper(paper["categories"], paper["abstract"])
        subfields_str = ",".join(subfields)
        # ",".join(subfields) — converts ["Computer Vision", "NLP"] to "Computer Vision,NLP" for storage. 
        # SQLite has no array type, so comma-separated strings are the clean solution here

        cursor.execute(
            "UPDATE papers SET subfields = ? WHERE id = ?",
            (subfields_str, paper["id"])
            # UPDATE papers SET subfields = ? WHERE id = ? — ? is a positional placeholder. SQLite replaces them with the 
            # tuple values (subfields_str, paper["id"]) in order. We use ? here instead of named placeholders 
            # because we only have two values
        )
        updated += 1

    conn.commit() # We commit once after all updates
    conn.close()
    print(f"Classification complete. Updated {updated} papers.")


def run_trend_analysis():
    conn   = get_connection()
    cursor = conn.cursor()

    papers = cursor.execute(
        "SELECT abstract, published_date FROM papers"
    ).fetchall()

    papers = [dict(p) for p in papers]
    # [dict(p) for p in papers] — sqlite3.Row objects are dict-like but not actual dicts. 
    # Converting them lets us use standard dict operations like .lower() chaining cleanly in compute_frequency

    historical, recent = split_papers_by_period(papers)
    print(f"Historical period: {len(historical)} papers")
    print(f"Recent period    : {len(recent)} papers")

    inserted = 0

    for keyword in TREND_KEYWORDS:
        hist_freq   = compute_frequency(historical, keyword)
        recent_freq = compute_frequency(recent, keyword)
        growth      = compute_growth_rate(hist_freq, recent_freq)
        status      = classify_trend_status(growth, recent_freq)

        for period, freq in [("historical", hist_freq), ("recent", recent_freq)]: # for period, freq in [("historical", hist_freq), ("recent", recent_freq)] — a clean way to insert two related rows without repeating the cursor.execute call twice. We loop over a list of tuples
            cursor.execute("""
                INSERT OR REPLACE INTO trends (keyword, period, frequency, growth_rate)
                VALUES (?, ?, ?, ?)
            """, (keyword, period, freq, growth))
            inserted += 1
            # INSERT OR REPLACE — unlike INSERT OR IGNORE, this overwrites the existing row if the unique constraint fires. 
            # Useful here because re-running trend analysis should update frequencies with fresh data, not skip them

    conn.commit()
    conn.close()
    print(f"Trend analysis complete. {inserted} trend records written.")


if __name__ == "__main__":
    run_classification()
    run_trend_analysis()