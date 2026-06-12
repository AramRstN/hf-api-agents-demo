"""
llm_api.py

Helper functions for calling cloud and local LLMs.

This file shows two approaches:

1. OpenAI cloud model
2. Ollama local model through OpenAI-compatible API
"""

from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

OLLAMA_BASE_URL = "http://localhost:11434/v1"

openai_client = OpenAI()

ollama_client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama"
)


def ask_openai(
    user_message: str,
    system_message: str = "You are a helpful teaching assistant.",
    model: str = "gpt-4.1-mini",
) -> str:
    """
    Send one message to an OpenAI model and return the answer.
    """

    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
    )

    return response.choices[0].message.content


def ask_ollama(
    user_message: str,
    system_message: str = "You are a helpful teaching assistant.",
    model: str = "llama3.2",
) -> str:
    """
    Send one message to a local Ollama model and return the answer.
    """

    response = ollama_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
    )

    return response.choices[0].message.content


def stream_openai(
    user_message: str,
    system_message: str = "You are a helpful teaching assistant.",
    model: str = "gpt-4.1-mini",
):
    """
    Stream an OpenAI response piece by piece.

    This is useful for Gradio apps where we want the answer
    to appear gradually.
    """

    stream = openai_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
        stream=True,
    )

    partial_answer = ""

    for chunk in stream:
        delta = chunk.choices[0].delta.content

        if delta is not None:
            partial_answer += delta
            yield partial_answer