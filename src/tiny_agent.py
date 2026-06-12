"""
tiny_agent.py

A very small rule-based agent example.

This is not a full production agent.
It is a teaching example to show the idea:

User request → decide which tool to use → call tool → return answer
"""

from src.llm_api import ask_openai


def calculator_tool(expression: str) -> str:
    """
    Very simple calculator tool.

    Warning:
    eval is dangerous in real applications.
    Here it is only used as a classroom toy example.
    """

    try:
        allowed_chars = "0123456789+-*/(). "
        if not all(char in allowed_chars for char in expression):
            return "Invalid expression. Only basic arithmetic is allowed."

        result = eval(expression)
        return str(result)

    except Exception as error:
        return f"Calculator error: {error}"


def text_tool(text: str) -> str:
    """
    Simple text tool that asks an LLM to explain something.
    """

    return ask_openai(
        user_message=text,
        system_message="You are a helpful teaching assistant. Explain briefly.",
    )


def tiny_agent(user_message: str) -> str:
    """
    Decide which tool to use.

    This simple agent uses rules:
    - If the message starts with 'calculate:', use calculator.
    - Otherwise, send it to an LLM.
    """

    if user_message.lower().startswith("calculate:"):
        expression = user_message.replace("calculate:", "").strip()
        return calculator_tool(expression)

    return text_tool(user_message)


if __name__ == "__main__":
    print(tiny_agent("calculate: 10 + 5 * 2"))
    print(tiny_agent("Explain what an API is in one paragraph."))