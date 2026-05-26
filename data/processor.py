import re
from .db import get_connection

# CONFIGURATION

# direct lookup table. arXiv category → human-readable subfield.
# researchers assigned these themselves on submission — most reliable signal we have
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

# fallback for when categories are ambiguous or too broad.
# cs.LG (Machine Learning) could be anything — keywords help narrow it down
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

# disambiguation rules. "sequence" appears in both NLP and bioinformatics — without this,
# a protein folding paper would incorrectly get tagged as NLP
EXCLUDE_MAP = {
    "NLP"                   : ["protein", "dna", "rna"],
    "Graph Neural Networks" : ["plot", "figure", "chart"],
    "Computer Vision"       : ["audio", "speech", "acoustic"],
}

# terms we track over time — chosen because they represent either established techniques
# or fast-emerging ones, making them good candidates for gap analysis
TREND_KEYWORDS = [
    "transformer", "diffusion model", "large language model",
    "reinforcement learning", "graph neural network", "contrastive learning",
    "federated learning", "knowledge distillation", "self-supervised",
    "zero-shot", "few-shot", "prompt engineering", "fine-tuning",
    "multimodal", "vision transformer", "generative model",
    "anomaly detection", "object detection", "image segmentation",
    "drug discovery", "protein folding", "autonomous driving",
]

# ML techniques we track — the "how" of research
METHODS = [
    "transformer",
    "diffusion model",
    "graph neural network",
    "reinforcement learning",
    "contrastive learning",
    "federated learning",
    "knowledge distillation",
    "self-supervised",
    "zero-shot",
    "few-shot",
    "large language model",
    "vision transformer",
    "generative model",
    "autoencoder",
    "random forest",
]

# application areas we track — the "where" of research
DOMAINS = [
    "Healthcare",
    "Cybersecurity",
    "Finance",
    "Robotics",
    "Bioinformatics",
    "Education",
    "Climate",
    "Legal",
    "Manufacturing",
    "Agriculture",
]

# since domains aren't in arXiv categories, we detect them from abstract text
DOMAIN_KEYWORDS = {
    "Healthcare"    : ["medical", "clinical", "healthcare", "disease", "diagnosis", "patient", "hospital"],
    "Cybersecurity" : ["intrusion", "malware", "security", "attack", "vulnerability", "threat", "network security"],
    "Finance"       : ["stock", "financial", "trading", "market", "portfolio", "banking", "cryptocurrency"],
    "Robotics"      : ["robot", "autonomous", "navigation", "manipulation", "control system", "drone"],
    "Bioinformatics": ["protein", "genomic", "dna", "rna", "molecular", "biological", "genome"],
    "Education"     : ["learning outcome", "student", "e-learning", "education", "tutoring", "academic"],
    "Climate"       : ["climate", "weather", "carbon", "emission", "environmental", "temperature forecast"],
    "Legal"         : ["legal", "law", "court", "contract", "compliance", "regulation", "judicial"],
    "Manufacturing" : ["manufacturing", "industrial", "factory", "production", "quality control", "defect"],
    "Agriculture"   : ["crop", "agriculture", "farming", "soil", "yield", "irrigation", "pest"],
}

# if fewer than this many papers exist for a method-domain pair, we call it a gap.
# tunable — raise for stricter gaps, lower for more gaps
GAP_THRESHOLD = 5


# CLASSIFICATION HELPERS

def classify_paper(categories: str, abstract: str) -> list[str]:
    subfields      = set()
    # set not list — a paper matching cs.LG in CATEGORY_MAP AND "machine learning" in KEYWORD_MAP
    # would get duplicated in a list. set handles this automatically
    abstract_lower = abstract.lower()

    for cat_code, subfield in CATEGORY_MAP.items():
        if cat_code in categories:
            # categories is a space-separated string like "cs.LG cs.CV stat.ML".
            # Python's `in` checks substring existence — fast and readable
            subfields.add(subfield)

    for subfield, keywords in KEYWORD_MAP.items():
        if subfield in subfields:
            # category mapping already assigned this subfield — keyword matching would be redundant
            continue

        has_keyword = any(kw in abstract_lower for kw in keywords)
        if not has_keyword:
            continue

        exclusions  = EXCLUDE_MAP.get(subfield, [])
        # .get with default [] — if no exclusions exist for this subfield, the loop below runs harmlessly
        is_excluded = any(ex in abstract_lower for ex in exclusions)
        # short-circuits on first match — stops checking once one exclusion is found
        if not is_excluded:
            subfields.add(subfield)

    return sorted(subfields) if subfields else ["General Machine Learning"]
    # sorted for consistent ordering — ensures "Computer Vision,NLP" not "NLP,Computer Vision" randomly each run


# TREND HELPERS

def split_papers_by_period(papers: list) -> tuple[list, list]:
    sorted_papers = sorted(papers, key=lambda p: p["published_date"])
    # YYYY-MM-DD strings sort correctly as plain strings — year is leftmost so lexicographic = chronological
    midpoint = len(sorted_papers) // 2
    return sorted_papers[:midpoint], sorted_papers[midpoint:]


def compute_frequency(papers: list, keyword: str) -> float:
    if not papers:
        return 0.0
    count = sum(1 for p in papers if keyword in p["abstract"].lower())
    return count / len(papers)
    # normalize by corpus size so two periods of different sizes are comparable.
    # result: 30 matches in 150 papers → 0.2 (20% of papers mention this keyword)


def compute_growth_rate(historical_freq: float, recent_freq: float) -> float:
    if historical_freq == 0:
        return 1.0 if recent_freq > 0 else 0.0
        # can't divide by zero. keyword emerging from nothing → treat as 100% growth (1.0).
        # keyword absent in both periods → no meaningful growth (0.0)
    return (recent_freq - historical_freq) / historical_freq


def classify_trend_status(growth_rate: float, recent_freq: float) -> str:
    if recent_freq < 0.01:
        return "obsolete"
        # fewer than 1% of recent papers mention this — essentially gone regardless of growth rate
    if growth_rate > 0.5:
        return "widening"
        # grew more than 50% — gap between supply and demand is increasing
    if growth_rate < -0.2:
        return "closing"
        # shrank more than 20% — research is catching up with demand
    return "stable"


# GAP DETECTION HELPERS

def detect_domains(abstract: str) -> list[str]:
    abstract_lower = abstract.lower()
    matched = []

    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(kw in abstract_lower for kw in keywords):
            matched.append(domain)

    return matched if matched else []
    # return empty list rather than a fallback — unlike subfield classification,
    # forcing a domain label on a domain-less paper would pollute the gap matrix


def detect_methods(abstract: str) -> list[str]:
    abstract_lower = abstract.lower()
    return [method for method in METHODS if method in abstract_lower]
    # method names are specific enough to match directly — "transformer" in an abstract
    # almost always means the architecture, not something else


def build_gap_matrix(papers: list) -> dict:
    matrix = {
        (method, domain): 0
        for method in METHODS
        for domain in DOMAINS
    }
    # initialize every pair at zero upfront — without this, pairs with zero papers
    # wouldn't exist in the dict at all, making gap detection harder downstream

    for paper in papers:
        abstract = paper["abstract"]
        methods  = detect_methods(abstract)
        domains  = detect_domains(abstract)

        for method in methods:
            for domain in domains:
                matrix[(method, domain)] += 1
                # a paper mentioning two methods and two domains increments four cells —
                # correct, it's genuine evidence for all those combinations

    return matrix


def classify_gap_status(count: int, method: str, trends_data: dict) -> str:
    if count == 0:
        return "unexplored"
    if count <= GAP_THRESHOLD:
        trend = trends_data.get(method, {}).get("growth_rate", 0)
        # chained .get() with defaults — if method isn't in trends, we get {}.
        # then .get("growth_rate", 0) gives us 0. no KeyError, no crash
        if trend > 0.3:
            return "widening"
        if trend < -0.2:
            return "closing"
        return "emerging"
    return "active"


# RUNNERS

def run_classification():
    conn   = get_connection()
    cursor = conn.cursor()

    papers = cursor.execute(
        "SELECT id, categories, abstract FROM papers WHERE subfields IS NULL"
    ).fetchall()
    # WHERE subfields IS NULL — only processes unclassified papers.
    # safe to re-run without double-processing, same principle as INSERT OR IGNORE in scraper

    print(f"Classifying {len(papers)} papers...")
    updated = 0

    for paper in papers:
        subfields     = classify_paper(paper["categories"], paper["abstract"])
        subfields_str = ",".join(subfields)
        # SQLite has no array type — comma-separated string is the clean storage solution.
        # ["Computer Vision", "NLP"] → "Computer Vision,NLP"

        cursor.execute(
            "UPDATE papers SET subfields = ? WHERE id = ?",
            (subfields_str, paper["id"])
        )
        updated += 1

    conn.commit()
    # commit once after all updates, not inside the loop —
    # committing 300 times would hit disk 300 times. one commit is far faster
    conn.close()
    print(f"Classification complete. Updated {updated} papers.")


def run_trend_analysis():
    conn   = get_connection()
    cursor = conn.cursor()

    papers = cursor.execute(
        "SELECT abstract, published_date FROM papers"
    ).fetchall()

    papers = [dict(p) for p in papers]
    # sqlite3.Row objects are dict-like but not actual dicts — converting lets us
    # use standard dict operations cleanly inside compute_frequency

    historical, recent = split_papers_by_period(papers)
    print(f"Historical period: {len(historical)} papers")
    print(f"Recent period    : {len(recent)} papers")

    inserted = 0

    for keyword in TREND_KEYWORDS:
        hist_freq   = compute_frequency(historical, keyword)
        recent_freq = compute_frequency(recent, keyword)
        growth      = compute_growth_rate(hist_freq, recent_freq)

        for period, freq in [("historical", hist_freq), ("recent", recent_freq)]:
            # loop over a list of tuples — inserts both rows without repeating cursor.execute twice
            cursor.execute("""
                INSERT OR REPLACE INTO trends (keyword, period, frequency, growth_rate)
                VALUES (?, ?, ?, ?)
            """, (keyword, period, freq, growth))
            # INSERT OR REPLACE not INSERT OR IGNORE — re-running should update frequencies
            # with fresh data, not silently skip them
            inserted += 1

    conn.commit()
    conn.close()
    print(f"Trend analysis complete. {inserted} trend records written.")


def run_gap_detection():
    conn   = get_connection()
    cursor = conn.cursor()

    papers = [dict(p) for p in cursor.execute(
        "SELECT abstract FROM papers"
    ).fetchall()]

    trends_raw = cursor.execute(
        "SELECT keyword, growth_rate FROM trends WHERE period = 'recent'"
    ).fetchall()

    trends_data = {
        row["keyword"]: {"growth_rate": row["growth_rate"]}
        for row in trends_raw
    }
    # reshape flat DB rows into a nested dict keyed by keyword —
    # makes trends_data["transformer"]["growth_rate"] possible in classify_gap_status

    print(f"Building gap matrix from {len(papers)} papers...")
    matrix  = build_gap_matrix(papers)
    inserted = 0

    for (method, domain), count in matrix.items():
        status = classify_gap_status(count, method, trends_data)

        is_gap     = count <= GAP_THRESHOLD
        feas_score = min(100, count * 10) if count > 0 else 20
        # heuristic: 0 papers → 20 (not zero — combination might still make sense),
        # 1 paper → 10, 10+ papers → capped at 100. easy to understand and tune
        feas_label = "High" if feas_score >= 70 else "Medium" if feas_score >= 40 else "Low"

        cursor.execute("""
            INSERT OR REPLACE INTO gaps
                (method, domain, paper_count, feasibility_score, feasibility_label, status)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (method, domain, count, feas_score, feas_label, status))
        # INSERT OR REPLACE — re-running updates scores rather than creating duplicates
        # because UNIQUE(method, domain) fires and REPLACE overwrites cleanly

        if is_gap:
            inserted += 1

    conn.commit()
    conn.close()
    print(f"Gap detection complete. {inserted} gaps identified.")


if __name__ == "__main__":
    run_classification()
    run_trend_analysis()
    run_gap_detection()