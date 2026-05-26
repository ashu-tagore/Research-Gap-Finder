from data.db        import init_db
from data.scraper   import run_scraper
from data.processor import (
    run_classification,
    run_difficulty_scoring,
    run_trend_analysis,
    run_gap_detection,
)


def main():
    init_db()
    run_scraper()
    run_classification()
    run_difficulty_scoring()
    run_trend_analysis()
    run_gap_detection()


if __name__ == "__main__":
    main()
