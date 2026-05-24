import feedparser  # feedparser — parses the Atom XML that arXiv returns
import requests
import time
try:
    from .db import get_connection
except ImportError:
    from db import get_connection

# Configurations
ARXIV_API_URL = "https://export.arxiv.org/api/query"
HEADERS = {"User-Agent": "ResearchGapFinder/1.0 (aswat0thama@gmail.com)"}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)

SEARCH_QUERY = "cat:cs.AI OR cat:cs.LG OR cat:cs.CV OR cat:cs.CL OR cat:cs.NE OR cat:cs.RO"

MAX_RESULTS = 500
BATCH_SIZE  = 25
SLEEP_TIME  = 15

CODE_INDICATORS = [
    "github.com",
    "code available",
    "implementation at",
    "open-source",
    "open source",
    "https://github",
    "code is available",
]

# Helpers
def has_code(abstract: str, comments:str) -> bool:
    text = (abstract + " " + comments).lower()
    return any(indicator in text for indicator in CODE_INDICATORS)

def parse_entry(entry) -> dict:
    arxiv_id = entry.id.split("/abs/")[-1]
    title = entry.title.replace("\n", " ").strip()
    abstract = entry.summary.replace("\n", " ").strip()
    authors   = ", ".join(a.name for a in entry.get("authors", []))
    categories = " ".join(t.term for t in entry.get("tags", []))
    published  = entry.published[:10]
    pdf_link   = next(
        (l.href for l in entry.get("links", []) if l.get("type") == "application/pdf"),
        ""
    )

    comments = getattr(entry, "arxiv_comment", "") or ""

    return {
        "arxiv_id"   : arxiv_id,
        "title"      : title,
        "abstract"   : abstract,
        "authors"    : authors,
        "categories" : categories,
        "published_date": published,
        "pdf_link"   : pdf_link,
        "comments"   : comments,
        "has_code"   : int(has_code(abstract, comments)),
    }

# Core Scraper

MAX_RETRIES = 3
RETRYABLE_STATUS = {429, 500, 502, 503, 504}

def fetch_batch(start: int) -> list:
    params = {
        "search_query": SEARCH_QUERY,
        "start": start,
        "max_results": BATCH_SIZE,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = SESSION.get(ARXIV_API_URL, params=params, timeout=60)
        except requests.exceptions.Timeout:
            error_msg = "timed out"
            wait = attempt * 20
        else:
            if response.status_code == 200:
                feed = feedparser.parse(response.text)
                return [parse_entry(e) for e in feed.entries]
            if response.status_code not in RETRYABLE_STATUS:
                print(f"  Error fetching batch at start={start}: HTTP {response.status_code}")
                return None
            error_msg = f"HTTP {response.status_code}"
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After", "")
                wait = int(retry_after) if retry_after.isdigit() else (2 ** attempt) * 60
            else:
                wait = attempt * 20

        if attempt < MAX_RETRIES:
            print(f"  Batch start={start} got {error_msg}, retrying in {wait}s (attempt {attempt}/{MAX_RETRIES})...")
            time.sleep(wait)
        else:
            print(f"  Batch start={start} failed after {MAX_RETRIES} attempts ({error_msg}) — skipping.")

    return None


def save_papers(papers: list) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    saved = 0

    for paper in papers:
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO papers (
                    arxiv_id, title, abstract, authors,
                    categories, published_date, pdf_link,
                    comments, has_code
                ) VALUES (
                    :arxiv_id, :title, :abstract, :authors,
                    :categories, :published_date, :pdf_link,
                    :comments, :has_code
                )
            """, paper)

            if cursor.rowcount == 1:
                saved += 1

        except Exception as e:
            print(f"  Skipping {paper['arxiv_id']}: {e}")
    
    conn.commit()
    conn.close()
    return saved

def run_scraper():
    print(f"Starting scrape: {MAX_RESULTS} papers in batches of {BATCH_SIZE}")
    total_saved = 0

    for start in range(0, MAX_RESULTS, BATCH_SIZE):
        batch_num = (start // BATCH_SIZE) + 1
        print(f"  Fetching batch {batch_num} (papers {start+1}-{start+BATCH_SIZE})...")

        papers = fetch_batch(start)

        if papers is None:
            print("  Batch failed — skipping.")
            continue

        if not papers:
            print("  Empty batch — stopping early.")
            break

        saved = save_papers(papers)
        total_saved += saved
        print(f"  Saved {saved} new papers.")

        if start + BATCH_SIZE < MAX_RESULTS:  #  we only sleep between batches, not after the last one
            time.sleep(SLEEP_TIME)

    print(f"\nScrape complete. Total new papers saved: {total_saved}")


if __name__ == "__main__":
    run_scraper()