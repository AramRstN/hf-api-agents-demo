"""
debate.py

Reusable logic for running a conversation between:
1. A cloud GPT model from OpenAI
2. A local model running through Ollama

This file does NOT launch a UI.
It only defines functions that can be imported by other files.
"""

from openai import OpenAI
from dotenv import load_dotenv


# ------------------------------------------------------------
# Setup
# ------------------------------------------------------------

load_dotenv()

OLLAMA_BASE_URL = "http://localhost:11434/v1"

gpt_client = OpenAI()

ollama_client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama"
)


# ------------------------------------------------------------
# Helper function for GPT
# ------------------------------------------------------------

def call_gpt(
    gpt_client,
    gpt_model,
    gpt_system,
    gpt_messages,
    ollama_messages
):
    """
    Call the GPT model using the conversation history.

    From GPT's perspective:
    - GPT's own previous messages are assistant messages.
    - Ollama's previous messages are user messages.
    """

    messages = [
        {
            "role": "system",
            "content": gpt_system
        }
    ]

    for gpt_message, ollama_message in zip(gpt_messages, ollama_messages):
        messages.append({
            "role": "assistant",
            "content": gpt_message
        })

        messages.append({
            "role": "user",
            "content": ollama_message
        })

    response = gpt_client.chat.completions.create(
        model=gpt_model,
        messages=messages
    )

    return response.choices[0].message.content


# ------------------------------------------------------------
# Helper function for Ollama
# ------------------------------------------------------------

def call_ollama(
    ollama_client,
    ollama_model,
    ollama_system,
    gpt_messages,
    ollama_messages
):
    """
    Call the Ollama model using the conversation history.

    From Ollama's perspective:
    - GPT's previous messages are user messages.
    - Ollama's own previous messages are assistant messages.
    """

    messages = [
        {
            "role": "system",
            "content": ollama_system
        }
    ]

    for gpt_message, ollama_message in zip(gpt_messages, ollama_messages):
        messages.append({
            "role": "user",
            "content": gpt_message
        })

        messages.append({
            "role": "assistant",
            "content": ollama_message
        })

    # GPT has just replied, so Ollama should respond to GPT's latest message
    messages.append({
        "role": "user",
        "content": gpt_messages[-1]
    })

    response = ollama_client.chat.completions.create(
        model=ollama_model,
        messages=messages
    )

    return response.choices[0].message.content


# ------------------------------------------------------------
# Main reusable debate function
# ------------------------------------------------------------

def run_debate(
    gpt_model="gpt-4.1-mini",
    ollama_model="llama3.2",
    gpt_name="Argumentative GPT",
    ollama_name="Polite Llama",
    gpt_system=None,
    ollama_system=None,
    topic="Should students use cloud LLM APIs or local open-source models?",
    number_of_rounds=3
):
    """
    Run a full debate and return the conversation as a formatted string.

    This function is designed to be called from:
    - a script,
    - a notebook,
    - a Gradio app.
    """

    if gpt_system is None:
        gpt_system = """
You are a chatbot who is very argumentative.
You disagree with almost everything in the conversation.
You challenge every idea in a snarky but not offensive way.
You should still stay helpful and technically meaningful.
Keep your answers short: maximum 4 sentences.
"""

    if ollama_system is None:
        ollama_system = """
You are a very polite and calm chatbot.
You try to find common ground with the other person.
If the other person is argumentative, you try to calm them down.
You should still give useful and thoughtful answers.
Keep your answers short: maximum 4 sentences.
"""

    gpt_messages = [
        f"Hi there. I want to debate this topic: {topic}"
    ]

    ollama_messages = [
        "Hello! I am happy to have a thoughtful conversation with you."
    ]

    debate_output = ""

    for round_number in range(int(number_of_rounds)):
        debate_output += "\n" + "=" * 60 + "\n"
        debate_output += f"Round {round_number + 1}\n"
        debate_output += "=" * 60 + "\n\n"

        # GPT responds to Ollama
        gpt_reply = call_gpt(
            gpt_client=gpt_client,
            gpt_model=gpt_model,
            gpt_system=gpt_system,
            gpt_messages=gpt_messages,
            ollama_messages=ollama_messages
        )

        gpt_messages.append(gpt_reply)

        debate_output += f"{gpt_name}:\n"
        debate_output += f"{gpt_reply}\n\n"

        # Ollama responds to GPT
        ollama_reply = call_ollama(
            ollama_client=ollama_client,
            ollama_model=ollama_model,
            ollama_system=ollama_system,
            gpt_messages=gpt_messages,
            ollama_messages=ollama_messages
        )

        ollama_messages.append(ollama_reply)

        debate_output += f"{ollama_name}:\n"
        debate_output += f"{ollama_reply}\n\n"

    return debate_output


# ------------------------------------------------------------
# Optional: allow this file to run directly
# ------------------------------------------------------------

if __name__ == "__main__":
    output = run_debate()
    print(output)