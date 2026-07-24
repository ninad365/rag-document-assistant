from app.rag.prompts import build_answer_prompt, build_rewrite_prompt


def test_rewrite_prompt_contains_history_and_question():
    prompt = build_rewrite_prompt(
        [{"role": "user", "content": "What is it?"}, {"role": "assistant", "content": "It is X."}],
        "How does it work?",
    )
    assert "How does it work?" in prompt
    assert "assistant: It is X." in prompt


def test_answer_prompt_has_guardrail():
    prompt = build_answer_prompt("ctx", "q")
    assert "Use only the provided context" in prompt
