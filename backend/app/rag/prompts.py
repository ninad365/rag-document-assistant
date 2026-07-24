from textwrap import dedent


def build_rewrite_prompt(history: list[dict], question: str) -> str:
    conversation = "\n".join(f"{m['role']}: {m['content']}" for m in history)
    return dedent(
        f"""
        Rewrite the user's latest question into a standalone question.
        Preserve factual intent and entities.

        Conversation history:
        {conversation}

        Latest question:
        {question}

        Return only the rewritten standalone question.
        """
    ).strip()


def build_answer_prompt(context: str, question: str) -> str:
    return dedent(
        f"""
        You are a retrieval-augmented assistant.
        Use only the provided context. If the answer is not in the context,
        answer exactly: "I could not find this in the uploaded documents."

        Context:
        {context}

        Question:
        {question}
        """
    ).strip()
