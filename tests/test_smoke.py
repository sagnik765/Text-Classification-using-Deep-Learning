from pathlib import Path


def test_core_files_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "Text_Classification_with_Deep_Learning.ipynb").exists()
    assert (root / "text_classification.tar.gz").exists()
