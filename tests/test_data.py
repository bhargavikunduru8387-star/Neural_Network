
from pathlib import Path

import pandas as pd


DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "iris_dataset.csv"
)


def test_dataset_exists():
    assert DATA_PATH.exists(), (
        f"Dataset not found: {DATA_PATH}"
    )



def test_dataset_columns():
    df = pd.read_csv(DATA_PATH)

    expected_columns = {
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
        "species",
    }

    assert expected_columns.issubset(
        set(df.columns)
    ), f"Unexpected columns: {list(df.columns)}"


def test_dataset_has_no_missing_values():
    df = pd.read_csv(DATA_PATH)

    assert not df.isnull().any().any(), (
        "Dataset contains missing values"
    )


def test_dataset_has_valid_species():
    df = pd.read_csv(DATA_PATH)

    expected_species = {
        "setosa",
        "versicolor",
        "virginica",
    }

    actual_species = set(
        df["species"].astype(str).str.strip().str.lower()
    )

    assert actual_species == expected_species
