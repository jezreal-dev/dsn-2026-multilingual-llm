"""Unit and integration test suite for Data Ingestion & Governance (Milestone 1)."""

import csv
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.data.loader import DatasetLoader
from src.data.schema import Language, NewsArticle, TopicLabel

DATA_DIR = Path("/home/jmomoh/dsn-ai-bootcamp/data")


class TestSchemaValidation:
    """Rigorous boundary and type checking for data contracts."""

    def test_valid_news_article(self) -> None:
        article = NewsArticle(
            id="hau_0001",
            lang="hau",
            text="Wannan labari ne mai dadi.",
            headline="Labari",
            category="health",
        )
        assert article.lang == Language.HAUSA
        assert article.category == TopicLabel.HEALTH
        assert article.lang.full_name == "Hausa"

    def test_invalid_category_raises(self) -> None:
        with pytest.raises(ValidationError):
            NewsArticle(
                id="yor_0001",
                lang="yor",
                text="Irohin ayo",
                headline="Irohin",
                category="invalid_category_123",
            )

    def test_invalid_language_raises(self) -> None:
        with pytest.raises(ValidationError):
            NewsArticle(
                id="eng_0001",
                lang="eng",
                text="English is not allowed",
                headline="English",
                category="sports",
            )

    def test_empty_text_raises(self) -> None:
        with pytest.raises(ValidationError):
            NewsArticle(
                id="ibo_0001",
                lang="ibo",
                text="   ",
                headline="Akuko",
                category="politics",
            )


class TestDatasetLoader:
    """Audit of file I/O, schema conformance, and dataset partition invariants."""

    @pytest.fixture
    def loader(self) -> DatasetLoader:
        return DatasetLoader(DATA_DIR)

    def test_train_split_invariants(self, loader: DatasetLoader) -> None:
        train_data = loader.load_train()
        assert len(train_data) == 6068, f"Expected 6,068 train rows, got {len(train_data)}"

        profile = loader.profile(train_data)
        assert profile["total_samples"] == 6068
        assert "languages" in profile
        assert "categories" in profile

        # Ensure all 4 languages exist in training
        langs = profile["languages"]
        assert set(langs.keys()) == {"hau", "ibo", "pcm", "yor"}

        # Ensure all 7 categories exist in training
        cats = profile["categories"]
        assert set(cats.keys()) == {
            "business", "health", "politics", "religion",
            "sports", "entertainment", "technology"
        }

    def test_dev_split_invariants(self, loader: DatasetLoader) -> None:
        dev_data = loader.load_dev()
        assert len(dev_data) == 869, f"Expected 869 dev rows, got {len(dev_data)}"

        profile = loader.profile(dev_data)
        langs = profile["languages"]
        assert langs["hau"]["count"] == 317
        assert langs["ibo"]["count"] == 194
        assert langs["pcm"]["count"] == 152
        assert langs["yor"]["count"] == 206

    def test_test_split_invariants(self, loader: DatasetLoader) -> None:
        test_data = loader.load_test()
        assert len(test_data) == 1743, f"Expected 1,743 test rows, got {len(test_data)}"

        # Ground truth should be withheld for test set
        for item in test_data:
            assert item.category is None
            assert item.headline is None

    def test_corrupted_row_handling(self, tmp_path: Path) -> None:
        corrupt_csv = tmp_path / "corrupt.csv"
        with corrupt_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "lang", "text", "headline", "category"])
            writer.writerow(["bad_1", "invalid_lang", "Some text", "A headline", "health"])

        loader = DatasetLoader(tmp_path)
        with pytest.raises(ValueError, match="Validation failure"):
            loader.load_split("corrupt.csv", is_test=False)
