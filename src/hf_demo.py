"""
hf_demo.py

Simple Hugging Face model reuse demo.

This file shows how to use a pretrained model directly
without training our own model from scratch.
"""

from transformers import pipeline


def create_sentiment_classifier():
    """
    Load a pretrained sentiment analysis pipeline.
    """

    classifier = pipeline("sentiment-analysis")
    return classifier


def classify_sentiment(text: str):
    """
    Classify the sentiment of a piece of text.
    """

    classifier = create_sentiment_classifier()
    result = classifier(text)
    return result


if __name__ == "__main__":
    example_text = "I really enjoyed this AI workshop!"
    prediction = classify_sentiment(example_text)

    print("Input:")
    print(example_text)

    print("\nPrediction:")
    print(prediction)