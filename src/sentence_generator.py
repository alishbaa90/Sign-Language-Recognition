import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def generate_sentence(sign_words):
    model = genai.GenerativeModel("gemini-3.6-flash")
    prompt = f"""
    These are individual sign language words detected in sequence: {sign_words}.
    Convert them into a single natural, grammatically correct English sentence
    that captures the person's actual intent. Return only the final sentence, nothing else.
    """
    response = model.generate_content(prompt)
    return response.text.strip()

if __name__ == "__main__":
    print(generate_sentence(["HELLO", "HOW", "YOU", "GOOD"]))
    # Expected: "Hello! How are you today?"