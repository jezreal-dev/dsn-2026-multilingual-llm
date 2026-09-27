"""Robust CSV data loading, schema enforcement, and dataset profiling."""

import csv
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List

from src.data.schema import Language, NewsArticle, TopicLabel

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Production-grade dataset loader enforcing schema validation and profiling."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Data directory does not exist: {self.data_dir}")

    def load_split(self, filename: str, is_test: bool = False) -> List[NewsArticle]:
        """Load and strictly validate a dataset split.

        Args:
            filename: Target CSV file name.
            is_test: True if loading test set (where headline and category are withheld).

        Returns:
            List of validated NewsArticle objects.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If schema violations or corrupted rows are detected.
        """
        filepath = self.data_dir / filename
        if not filepath.is_file():
            raise FileNotFoundError(f"Target split file not found: {filepath}")

        articles: List[NewsArticle] = []
        with filepath.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"CSV file is empty or corrupted: {filepath}")

            for row_idx, raw_row in enumerate(reader, start=1):
                try:
                    raw_cat = raw_row.get("category")
                    cat_val = TopicLabel.from_str(raw_cat) if raw_cat else None
                    lang_val = Language(raw_row["lang"].strip().lower())

                    article = NewsArticle(
                        id=raw_row["id"],
                        lang=lang_val,
                        text=raw_row["text"],
                        headline=raw_row.get("headline"),
                        category=cat_val,
                        url=raw_row.get("url"),
                        split=raw_row.get("split"),
                    )
                    if not is_test:
                        if article.category is None:
                            raise ValueError(f"Missing category in labeled split at row {row_idx}")
                        if article.headline is None:
                            raise ValueError(f"Missing headline in labeled split at row {row_idx}")
                    articles.append(article)
                except Exception as exc:
                    raise ValueError(f"Validation failure in {filename} at row {row_idx}: {exc}") from exc

        logger.info("Loaded %d validated articles from %s", len(articles), filename)
        return articles

    def load_train(self) -> List[NewsArticle]:
        """Load training split."""
        return self.load_split("train.csv", is_test=False)

    def load_dev(self) -> List[NewsArticle]:
        """Load validation split."""
        return self.load_split("dev.csv", is_test=False)

    def load_test(self) -> List[NewsArticle]:
        """Load test split."""
        return self.load_split("test.csv", is_test=True)

    @staticmethod
    def profile(articles: List[NewsArticle]) -> Dict[str, object]:
        """Compute statistical breakdown of languages and categories.

        Time Complexity: O(N) where N is number of articles.
        Space Complexity: O(L + C) where L is unique languages and C is unique categories.
        """
        total = len(articles)
        lang_counts = Counter(a.lang.value for a in articles)
        cat_counts = Counter(a.category.value for a in articles if a.category is not None)

        lang_breakdown = {k: {"count": v, "pct": round(v / total * 100, 2)} for k, v in lang_counts.items()}
        cat_breakdown = {k: {"count": v, "pct": round(v / total * 100, 2)} for k, v in cat_counts.items()} if cat_counts else {}

        return {
            "total_samples": total,
            "languages": lang_breakdown,
            "categories": cat_breakdown,
        }
