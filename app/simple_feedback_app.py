import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from pathlib import Path

model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
classifier = pipeline("sentiment-analysis", model=model_name)

CSV_FILE = Path("feedback_history.csv")
LABELS = ["positive", "neutral", "negative"]


def empty_df():
    return pd.DataFrame(columns=["feedback", "label", "confidence"])


def load_feedback():
    if CSV_FILE.exists():
        return pd.read_csv(CSV_FILE)
    return empty_df()


def save_feedback(df):
    df.to_csv(CSV_FILE, index=False)


def make_chart(df):
    counts = df["label"].value_counts().reindex(LABELS, fill_value=0)

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.bar(
        LABELS,
        [counts["positive"], counts["neutral"], counts["negative"]],
        color=["green", "gray", "red"]
    )

    ax.set_title("Feedback Sentiment Distribution")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of feedback")
    ax.set_ylim(0, max(1, counts.max() + 1))

    for i, value in enumerate(counts):
        ax.text(i, value + 0.05, str(value), ha="center")

    plt.tight_layout()
    return fig


def make_summary(df):
    counts = df["label"].value_counts().reindex(LABELS, fill_value=0)

    total = len(df)

    if total == 0:
        positivity = 0
    else:
        positivity = round((counts["positive"] / total) * 100, 1)

    return f"""
### Class Summary

Total feedback: **{total}**

Positive: **{counts["positive"]}**  
Neutral: **{counts["neutral"]}**  
Negative: **{counts["negative"]}**

Average positivity: **{positivity}%**
"""


def classify_feedback(feedback_text):
    df = load_feedback()

    if not feedback_text.strip():
        return "Please write one sentence.", df, make_chart(df), make_summary(df), ""

    prediction = classifier(feedback_text)[0]

    new_row = pd.DataFrame([{
        "feedback": feedback_text,
        "label": prediction["label"],
        "confidence": round(prediction["score"], 3)
    }])

    df = pd.concat([df, new_row], ignore_index=True)
    save_feedback(df)

    return (
        f"Prediction: **{prediction['label']}**",
        df,
        make_chart(df),
        make_summary(df),
        ""
    )


def reset_feedback():
    if CSV_FILE.exists():
        CSV_FILE.unlink()

    df = empty_df()

    return (
        "Feedback history cleared.",
        df,
        make_chart(df),
        make_summary(df),
        ""
    )


with gr.Blocks() as demo:
    gr.Markdown("# Live Lecture Feedback Classifier")

    feedback_input = gr.Textbox(
        label="Your feedback",
        placeholder="Write one sentence about today's class..."
    )

    submit_btn = gr.Button("Submit feedback")
    reset_btn = gr.Button("Reset feedback history")

    result_text = gr.Markdown()

    feedback_table = gr.Dataframe(
        label="All feedback history",
        value=empty_df()
    )

    feedback_chart = gr.Plot(
        label="Feedback chart",
        value=make_chart(empty_df())
    )

    summary_text = gr.Markdown(make_summary(empty_df()))

    submit_btn.click(
        fn=classify_feedback,
        inputs=feedback_input,
        outputs=[
            result_text,
            feedback_table,
            feedback_chart,
            summary_text,
            feedback_input
        ]
    )

    reset_btn.click(
        fn=reset_feedback,
        inputs=None,
        outputs=[
            result_text,
            feedback_table,
            feedback_chart,
            summary_text,
            feedback_input
        ]
    )


demo.launch(share=True)