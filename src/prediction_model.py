"""
prediction_model.py

Simple Hugging Face pipeline demo for heart disease prediction.

This script:
1. Loads a local heart_disease.csv dataset
2. Converts each row into a text description
3. Uses a Hugging Face zero-shot classification pipeline
4. Predicts whether the case looks like:
   - "heart disease present"
   - "no heart disease"

Important:
This is only an educational demo for model reuse.
It is not a real medical diagnosis system.
"""

from pathlib import Path

import pandas as pd
from transformers import pipeline


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

DATA_PATH = "data/heart_disease.csv"
OUTPUT_PATH = "data/heart_disease_predictions.csv"

MODEL_NAME = "facebook/bart-large-mnli"

CANDIDATE_LABELS = [
    "heart disease present",
    "no heart disease",
]


# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------

def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load the heart disease dataset.
    """

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find dataset at: {csv_path}"
        )

    return pd.read_csv(csv_path)


# ------------------------------------------------------------
# Convert one row to text
# ------------------------------------------------------------

def row_to_text(row: pd.Series) -> str:
    """
    Convert one patient row into a natural language description.

    This allows us to use a text-based transformer pipeline
    on tabular data.
    """

    return (
        f"Patient information: "
        f"age is {row.get('age')}, "
        f"sex is {row.get('sex')}, "
        f"chest pain type is {row.get('cp')}, "
        f"resting blood pressure is {row.get('trestbps')}, "
        f"cholesterol is {row.get('chol')}, "
        f"fasting blood sugar is {row.get('fbs')}, "
        f"resting ECG result is {row.get('restecg')}, "
        f"maximum heart rate is {row.get('thalach')}, "
        f"exercise induced angina is {row.get('exang')}, "
        f"oldpeak is {row.get('oldpeak')}, "
        f"slope is {row.get('slope')}, "
        f"number of major vessels is {row.get('ca')}, "
        f"thalassemia value is {row.get('thal')}."
    )


# ------------------------------------------------------------
# Create predictor
# ------------------------------------------------------------

def create_predictor():
    """
    Create a Hugging Face zero-shot classification pipeline.
    """

    classifier = pipeline(
        task="zero-shot-classification",
        model=MODEL_NAME,
    )

    return classifier


# ------------------------------------------------------------
# Predict heart disease cases
# ------------------------------------------------------------

def predict_heart_disease_cases(
    csv_path: str = DATA_PATH,
    output_path: str = OUTPUT_PATH,
    max_rows: int | None = 10,
) -> pd.DataFrame:
    """
    Apply the Hugging Face pipeline to the dataset.

    max_rows is used to keep the demo fast.
    Set max_rows=None if you want to process the full dataset.
    """

    df = load_dataset(csv_path)

    if max_rows is not None:
        df = df.head(max_rows).copy()

    classifier = create_predictor()

    predictions = []
    scores = []
    input_texts = []

    for _, row in df.iterrows():
        text = row_to_text(row)

        result = classifier(
            text,
            candidate_labels=CANDIDATE_LABELS,
        )

        predicted_label = result["labels"][0]
        predicted_score = result["scores"][0]

        input_texts.append(text)
        predictions.append(predicted_label)
        scores.append(predicted_score)

    result_df = df.copy()
    result_df["input_text_for_model"] = input_texts
    result_df["predicted_label"] = predictions
    result_df["prediction_score"] = scores

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result_df.to_csv(output_path, index=False)

    return result_df


# ------------------------------------------------------------
# Run as script
# ------------------------------------------------------------

if __name__ == "__main__":
    predictions_df = predict_heart_disease_cases(
        csv_path=DATA_PATH,
        output_path=OUTPUT_PATH,
        max_rows=10,
    )

    print("Predictions completed.")
    print(f"Saved results to: {OUTPUT_PATH}")
    print()
    print(predictions_df[["predicted_label", "prediction_score"]].head())