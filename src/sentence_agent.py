from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
import google.generativeai as genai
import os
from dotenv import load_dotenv
from logger_config import get_logger

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
logger = get_logger("sentence_agent")

model = genai.GenerativeModel("gemini-3.6-flash")


class AgentState(TypedDict):
    raw_signs: List[str]
    avg_confidence: float
    cleaned_signs: List[str]
    english_sentence: str
    toned_sentence: str
    german_sentence: str
    low_confidence_flag: bool


def cleanup_node(state: AgentState) -> AgentState:
    raw = state["raw_signs"]
    logger.info(f"Cleaning up raw signs: {raw}")
    prompt = f"""
    These are raw sign-language word predictions, possibly with noise or repeats: {raw}.
    Return ONLY a clean Python-style list of the most likely intended words, in order,
    removing obvious duplicates or noise. Return just the list, nothing else.
    Example output format: ["HELLO", "YOU", "GOOD"]
    """
    response = model.generate_content(prompt)
    cleaned_text = response.text.strip().strip("[]").replace('"', '').replace("'", "")
    cleaned = [w.strip() for w in cleaned_text.split(",") if w.strip()]
    logger.info(f"Cleaned signs: {cleaned}")
    return {"cleaned_signs": cleaned}


def confidence_check_node(state: AgentState) -> AgentState:
    avg_conf = state["avg_confidence"]
    flag = avg_conf < 0.45  # threshold below which we treat this as "low confidence"
    if flag:
        logger.warning(f"Low average confidence detected ({avg_conf:.2f}) — will hedge the sentence.")
    return {"low_confidence_flag": flag}


def grammar_node(state: AgentState) -> AgentState:
    signs = state["cleaned_signs"]
    low_conf = state["low_confidence_flag"]

    if low_conf:
        prompt = f"""
        These are individual sign language words detected in sequence, but detection confidence was LOW: {signs}.
        Convert them into a natural English sentence, but phrase it as a tentative best-guess
        (e.g. starting with "It looks like you might be saying..." or similar hedge).
        Return only the final sentence, nothing else.
        """
    else:
        prompt = f"""
        These are individual sign language words detected in sequence: {signs}.
        Convert them into a single natural, grammatically correct English sentence
        that captures the person's actual intent. Return only the final sentence, nothing else.
        """
    response = model.generate_content(prompt)
    logger.info(f"Generated English sentence: {response.text.strip()}")
    return {"english_sentence": response.text.strip()}


def tone_node(state: AgentState) -> AgentState:
    sentence = state["english_sentence"]
    prompt = f"""
    Rewrite this sentence to sound natural, warm, and conversational,
    as if a person genuinely said it in real life. Keep the meaning exactly the same.
    Return only the rewritten sentence, nothing else.
    Sentence: "{sentence}"
    """
    response = model.generate_content(prompt)
    logger.info(f"Toned sentence: {response.text.strip()}")
    return {"toned_sentence": response.text.strip()}


def translate_node(state: AgentState) -> AgentState:
    sentence = state["toned_sentence"]
    prompt = f"""
    Translate this sentence to natural German. Return only the translation, nothing else.
    Sentence: "{sentence}"
    """
    response = model.generate_content(prompt)
    logger.info(f"German translation: {response.text.strip()}")
    return {"german_sentence": response.text.strip()}


def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("cleanup", cleanup_node)
    graph.add_node("confidence_check", confidence_check_node)
    graph.add_node("grammar", grammar_node)
    graph.add_node("tone", tone_node)
    graph.add_node("translate", translate_node)

    graph.add_edge(START, "cleanup")
    graph.add_edge("cleanup", "confidence_check")
    graph.add_edge("confidence_check", "grammar")
    graph.add_edge("grammar", "tone")
    graph.add_edge("tone", "translate")
    graph.add_edge("translate", END)

    return graph.compile()


agent = build_agent()


def run_sentence_agent(raw_signs: List[str], avg_confidence: float = 1.0) -> dict:
    logger.info(f"Starting sentence agent run with {len(raw_signs)} signs, avg_confidence={avg_confidence:.2f}")
    result = agent.invoke({"raw_signs": raw_signs, "avg_confidence": avg_confidence})
    logger.info("Sentence agent run complete.")
    return result


if __name__ == "__main__":
    test_signs = ["HELLO", "HELLO", "YOU", "GOOD", "NO", "THANKYOU"]
    result = run_sentence_agent(test_signs, avg_confidence=0.35)  # simulate low confidence
    print("English (toned):", result["toned_sentence"])
    print("Low confidence flag:", result["low_confidence_flag"])