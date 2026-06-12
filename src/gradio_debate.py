"""
gradio_debate_solution.py

Reference solution for the Gradio exercise.

This app imports the debate logic from debate.py and builds a simple
Gradio UI around it.

"""

import gradio as gr

from debate import run_debate


# ------------------------------------------------------------
# Gradio wrapper function
# ------------------------------------------------------------

def run_debate_from_ui(
    gpt_model: str,
    ollama_model: str,
    gpt_name: str,
    ollama_name: str,
    gpt_system: str,
    ollama_system: str,
    topic: str,
    number_of_rounds: int,
) -> str:
    """
    Function called by the Gradio UI.

    It receives values from the interface, passes them to run_debate(),
    and returns the formatted debate output.
    """

    try:
        debate_output = run_debate(
            gpt_model=gpt_model,
            ollama_model=ollama_model,
            gpt_name=gpt_name,
            ollama_name=ollama_name,
            gpt_system=gpt_system,
            ollama_system=ollama_system,
            topic=topic,
            number_of_rounds=int(number_of_rounds),
        )

        return debate_output

    except Exception as error:
        return (
            "Something went wrong.\n\n"
            "Check that:\n"
            "1. Your OPENAI_API_KEY is available in the .env file.\n"
            "2. Ollama is running.\n"
            "3. The selected Ollama model is downloaded.\n"
            "4. The model names are correct.\n\n"
            f"Error details:\n{error}"
        )


# ------------------------------------------------------------
# Default values for the UI
# ------------------------------------------------------------

DEFAULT_GPT_SYSTEM = """You are a chatbot who is very argumentative.
You disagree with almost everything in the conversation.
You challenge every idea in a snarky but not offensive way.
You should still stay helpful and technically meaningful.
Keep your answers short: maximum 4 sentences."""

DEFAULT_OLLAMA_SYSTEM = """You are a very polite and calm chatbot.
You try to find common ground with the other person.
If the other person is argumentative, you try to calm them down.
You should still give useful and thoughtful answers.
Keep your answers short: maximum 4 sentences."""


# ------------------------------------------------------------
# Gradio UI
# ------------------------------------------------------------

with gr.Blocks(title="LLM Debate App") as demo:
    gr.Markdown("# LLM Debate App")
    gr.Markdown(
        """
Design two LLM personalities, choose a topic, and let them debate.

This app connects:

```text
Gradio UI → Python function → GPT model + Ollama model → Debate output
```

The important idea is that the models do not remember previous messages by themselves.
The debate logic stores the conversation history and sends it again in every model call.
"""
    )

    with gr.Row():
        gpt_model = gr.Textbox(
            label="OpenAI GPT Model",
            value="gpt-4.1-mini",
            placeholder="Example: gpt-4.1-mini",
        )

        ollama_model = gr.Textbox(
            label="Ollama Model",
            value="llama3.2",
            placeholder="Example: llama3.2",
        )

    with gr.Row():
        gpt_name = gr.Textbox(
            label="GPT Character Name",
            value="Argumentative GPT",
        )

        ollama_name = gr.Textbox(
            label="Ollama Character Name",
            value="Polite Llama",
        )

    with gr.Row():
        gpt_system = gr.Textbox(
            label="GPT System Message",
            value=DEFAULT_GPT_SYSTEM,
            lines=8,
        )

        ollama_system = gr.Textbox(
            label="Ollama System Message",
            value=DEFAULT_OLLAMA_SYSTEM,
            lines=8,
        )

    topic = gr.Textbox(
        label="Debate Topic",
        value="Should students use cloud LLM APIs or local open-source models?",
        lines=2,
    )

    number_of_rounds = gr.Slider(
        label="Number of Rounds",
        minimum=1,
        maximum=5,
        value=3,
        step=1,
    )

    start_button = gr.Button("Start Debate", variant="primary")

    debate_output = gr.Textbox(
        label="Debate Output",
        lines=25,
        show_copy_button=True,
    )

    start_button.click(
        fn=run_debate_from_ui,
        inputs=[
            gpt_model,
            ollama_model,
            gpt_name,
            ollama_name,
            gpt_system,
            ollama_system,
            topic,
            number_of_rounds,
        ],
        outputs=debate_output,
    )


# ------------------------------------------------------------
# Launch app
# ------------------------------------------------------------

if __name__ == "__main__":
    demo.launch()

    # For a temporary public link, use this instead:
    # demo.launch(share=True)
