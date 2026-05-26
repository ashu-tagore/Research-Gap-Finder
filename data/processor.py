import re
from .db import get_connection

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------

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

# difficulty signal weights — arbitrary but tunable.
# keyword complexity weighted highest because it's the most direct signal
# of theoretical depth in ML papers
DIFFICULTY_WEIGHTS = {
    "keyword"      : 0.35,
    "length"       : 0.15,
    "prerequisites": 0.30,
    "resources"    : 0.20,
}

# words that strongly indicate theoretical or mathematical depth
COMPLEXITY_KEYWORDS = [
    "theorem", "proof", "convergence", "optimization landscape",
    "gradient descent", "bayesian", "variational inference",
    "markov chain", "stochastic", "topology", "manifold",
    "eigenvalue", "matrix decomposition", "information theory",
]

# phrases that signal the paper builds on advanced prior work —
# a reader needs deep domain knowledge just to understand the baseline
PREREQUISITE_KEYWORDS = [
    "assumes familiarity",
    "building on",
    "extending",
    "following",
    "we adopt",
    "based on the framework",
    "novel framework",
    "first work to",
]

# phrases that signal expensive compute requirements —
# high resource need = high barrier to reproduce or build on
RESOURCE_KEYWORDS = [
    "v100", "a100", "h100",
    "gpu hours", "tpu",
    "billion parameters",
    "million samples",
    "weeks of training",
    "large-scale",
    "distributed training",
]

# methods with known successors — low paper count + successor exists = likely obsolete,
# not a genuine gap worth pursuing
SUCCESSOR_MAP = {
    "random forest"  : "gradient boosting or deep learning",
    "autoencoder"    : "diffusion model or VAE",
    "self-supervised": "contrastive learning or masked autoencoders",
}

# minimum papers across ALL domains for a method to be considered still active.
# if a method barely appears anywhere in recent literature, it's fading out
MIN_METHOD_ACTIVITY = 10

# minimum papers across ALL methods for a domain to be considered active research area.
# a domain with almost no papers is too niche to reliably detect gaps in
MIN_DOMAIN_ACTIVITY = 15

# if fewer than this many papers exist for a method-domain pair, we call it a gap.
# tunable — raise for stricter gaps, lower for more gaps
GAP_THRESHOLD = 5


# -------------------------------------------------------------------
# CLASSIFICATION HELPERS
# -------------------------------------------------------------------

def classify_paper(categories: str, abstract: str) -> list[str]:
    subfields      = set()
    # set not list — a paper matching cs.LG in CATEGORY_MAP AND "machine learning" in KEYWORD_MAP
    # would get duplicated in a list. set handles this automatically
    abstract_lower  = (abstract or "").lower()
    category_tokens = set((categories or "").split())

    for cat_code, subfield in CATEGORY_MAP.items():
        if cat_code in category_tokens or any(
            tok.startswith(cat_code + ".") for tok in category_tokens
        ):
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


# -------------------------------------------------------------------
# DIFFICULTY HELPERS
# -------------------------------------------------------------------

def score_keyword_complexity(abstract: str) -> float:
    abstract_lower = abstract.lower()
    matches        = sum(1 for kw in COMPLEXITY_KEYWORDS if kw in abstract_lower)
    return min(100.0, matches * 15.0)
    # each complexity keyword adds 15 points, capped at 100.
    # cap prevents a single unusually math-heavy abstract from dominating


def score_length(abstract: str) -> float:
    word_count = len(abstract.split())
    if word_count < 100:
        return 20.0
    if word_count < 150:
        return 40.0
    if word_count < 200:
        return 60.0
    return 80.0
    # longer abstracts tend to describe more complex systems —
    # simple papers say less, complex papers need more words to set up context


def score_prerequisites(abstract: str) -> float:
    abstract_lower = abstract.lower()
    matches        = sum(1 for kw in PREREQUISITE_KEYWORDS if kw in abstract_lower)
    return min(100.0, matches * 25.0)
    # prerequisite phrases are strong signals — even one match meaningfully
    # raises the bar for a reader, so each one gets a larger weight than complexity keywords


def score_resources(abstract: str) -> float:
    abstract_lower = abstract.lower()
    matches        = sum(1 for kw in RESOURCE_KEYWORDS if kw in abstract_lower)
    return min(100.0, matches * 30.0)
    # resource keywords are rare but decisive — mentioning "A100" or "billion parameters"
    # immediately signals this paper is out of reach for most researchers


def compute_difficulty(abstract: str) -> tuple[float, str]:
    abstract = abstract or ""
    scores = {
        "keyword"      : score_keyword_complexity(abstract),
        "length"       : score_length(abstract),
        "prerequisites": score_prerequisites(abstract),
        "resources"    : score_resources(abstract),
    }

    composite = sum(
        scores[signal] * weight
        for signal, weight in DIFFICULTY_WEIGHTS.items()
    )
    # weighted sum — each signal score (0-100) multiplied by its weight.
    # weights sum to 1.0 so composite stays in 0-100 range

    if composite < 25:
        level = "Beginner"
    elif composite < 50:
        level = "Intermediate"
    elif composite < 75:
        level = "Advanced"
    else:
        level = "Expert"

    return round(composite, 2), level
    # round to 2 decimal places — more precision implies false accuracy for a heuristic system


# -------------------------------------------------------------------
# TREND HELPERS
# -------------------------------------------------------------------

def split_papers_by_period(papers: list) -> tuple[list, list]:
    sorted_papers = sorted(papers, key=lambda p: p["published_date"])
    # YYYY-MM-DD strings sort correctly as plain strings — year is leftmost so lexicographic = chronological
    midpoint = len(sorted_papers) // 2
    return sorted_papers[:midpoint], sorted_papers[midpoint:]


def compute_frequency(papers: list, keyword: str) -> float:
    if not papers:
        return 0.0
    count = sum(1 for p in papers if keyword in (p["abstract"] or "").lower())
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


# -------------------------------------------------------------------
# GAP DETECTION HELPERS
# -------------------------------------------------------------------

def detect_domains(abstract: str) -> list[str]:
    abstract_lower = (abstract or "").lower()
    matched        = []

    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(kw in abstract_lower for kw in keywords):
            matched.append(domain)

    return matched if matched else []
    # return empty list rather than a fallback — unlike subfield classification,
    # forcing a domain label on a domain-less paper would pollute the gap matrix


def detect_methods(abstract: str) -> list[str]:
    abstract_lower = (abstract or "").lower()
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


# -------------------------------------------------------------------
# FEASIBILITY HELPERS
# -------------------------------------------------------------------

def check_obsolescence(method: str, method_total: int) -> tuple[bool, str]:
    is_too_rare = method_total < MIN_METHOD_ACTIVITY

    if is_too_rare and method in SUCCESSOR_MAP:
        reason = f"Likely replaced by {SUCCESSOR_MAP[method]}"
        return True, reason

    if is_too_rare:
        reason = f"Method appears in only {method_total} papers recently — may be fading"
        return True, reason

    return False, ""
    # returning (is_obsolete, reason) keeps the caller clean —
    # it gets both the decision and the explanation in one call


def compute_feasibility(
    count        : int,
    method_total : int,
    domain_total : int,
    is_obsolete  : bool,
) -> tuple[float, str]:

    score = 100.0

    if count == 0:
        score -= 40.0
        # zero papers means unverified — the combination might be nonsensical.
        # we don't go to zero because "unexplored" can still be valid

    if method_total < MIN_METHOD_ACTIVITY:
        score -= 35.0
        # method is barely used anywhere — high risk the gap exists because
        # researchers already moved on, not because it's an open problem

    if domain_total < MIN_DOMAIN_ACTIVITY:
        score -= 25.0
        # domain is too niche — even if you publish, audience is tiny
        # and finding collaborators or datasets will be hard

    if is_obsolete:
        score -= 20.0
        # stacks on top of method_total penalty — double confirmation of obsolescence

    score = max(0.0, score)
    # floor at zero — negative scores have no meaningful interpretation

    if score >= 70:
        label = "High"
    elif score >= 40:
        label = "Medium"
    else:
        label = "Low"

    return round(score, 2), label


# -------------------------------------------------------------------
# RUNNERS
# -------------------------------------------------------------------

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


def run_difficulty_scoring():
    conn   = get_connection()
    cursor = conn.cursor()

    papers = cursor.execute(
        "SELECT id, abstract FROM papers WHERE difficulty_score IS NULL"
    ).fetchall()
    # WHERE difficulty_score IS NULL — only scores unscored papers.
    # same idempotent pattern used throughout: safe to re-run, never double-processes

    print(f"Scoring difficulty for {len(papers)} papers...")
    updated = 0

    for paper in papers:
        score, level = compute_difficulty(paper["abstract"])

        cursor.execute(
            "UPDATE papers SET difficulty_score = ?, difficulty_level = ? WHERE id = ?",
            (score, level, paper["id"])
        )
        updated += 1

    conn.commit()
    conn.close()
    print(f"Difficulty scoring complete. Updated {updated} papers.")


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
        hist_freq    = compute_frequency(historical, keyword)
        recent_freq  = compute_frequency(recent, keyword)
        growth       = compute_growth_rate(hist_freq, recent_freq)
        trend_status = classify_trend_status(growth, recent_freq)

        for period, freq in [("historical", hist_freq), ("recent", recent_freq)]:
            status_for_row = trend_status if period == "recent" else None
            cursor.execute("""
                INSERT OR REPLACE INTO trends (keyword, period, frequency, growth_rate, trend_status)
                VALUES (?, ?, ?, ?, ?)
            """, (keyword, period, freq, growth, status_for_row))
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
    matrix = build_gap_matrix(papers)

    method_totals = {
        method: sum(matrix[(method, domain)] for domain in DOMAINS)
        for method in METHODS
    }
    domain_totals = {
        domain: sum(matrix[(method, domain)] for method in METHODS)
        for domain in DOMAINS
    }

    inserted = 0

    for (method, domain), count in matrix.items():
        status = classify_gap_status(count, method, trends_data)

        is_obsolete, obs_reason = check_obsolescence(method, method_totals[method])
        feas_score, feas_label  = compute_feasibility(
            count,
            method_totals[method],
            domain_totals[domain],
            is_obsolete,
        )

        flags  = obs_reason if is_obsolete else ""
        status = "obsolete" if is_obsolete else status

        cursor.execute("""
            INSERT OR REPLACE INTO gaps
                (method, domain, paper_count, feasibility_score, feasibility_label, status, flags)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (method, domain, count, feas_score, feas_label, status, flags))

        if count <= GAP_THRESHOLD:
            inserted += 1

    conn.commit()
    conn.close()
    print(f"Gap detection complete. {inserted} true gaps found (of {len(matrix)} total method-domain pairs).")


def run_feasibility_filter():
    # Standalone re-scoring utility — re-applies feasibility logic to existing gap rows
    # without rebuilding the full matrix. Useful when tuning thresholds (GAP_THRESHOLD,
    # MIN_METHOD_ACTIVITY, MIN_DOMAIN_ACTIVITY) without re-running the full pipeline.
    # Not called by the main pipeline because run_gap_detection already covers it.
    conn   = get_connection()
    cursor = conn.cursor()

    method_totals = {
        row["method"]: row["total"]
        for row in cursor.execute("""
            SELECT method, SUM(paper_count) as total
            FROM gaps
            GROUP BY method
        """).fetchall()
    }

    domain_totals = {
        row["domain"]: row["total"]
        for row in cursor.execute("""
            SELECT domain, SUM(paper_count) as total
            FROM gaps
            GROUP BY domain
        """).fetchall()
    }

    all_entries = cursor.execute(
        "SELECT id, method, domain, paper_count, status FROM gaps"
    ).fetchall()

    print(f"Running feasibility filter on {len(all_entries)} method-domain pairs...")
    updated = 0

    for gap in all_entries:
        method_total = method_totals.get(gap["method"], 0)
        domain_total = domain_totals.get(gap["domain"], 0)

        is_obsolete, obs_reason = check_obsolescence(gap["method"], method_total)

        feas_score, feas_label = compute_feasibility(
            gap["paper_count"],
            method_total,
            domain_total,
            is_obsolete,
        )

        flags  = obs_reason if is_obsolete else ""
        status = "obsolete" if is_obsolete else gap["status"]
        # override status to obsolete only when check_obsolescence confirms it —
        # preserves widening/closing/emerging for genuinely active gaps

        cursor.execute("""
            UPDATE gaps
            SET feasibility_score = ?,
                feasibility_label = ?,
                flags             = ?,
                status            = ?
            WHERE id = ?
        """, (feas_score, feas_label, flags, status, gap["id"]))
        updated += 1

    conn.commit()
    conn.close()
    print(f"Feasibility filter complete. Updated {updated} entries.")


if __name__ == "__main__":
    run_classification()
    run_difficulty_scoring()
    run_trend_analysis()
    run_gap_detection()