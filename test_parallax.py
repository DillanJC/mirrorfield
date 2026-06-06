"""
Parallax Pipeline Diagnostic Test
Tests individual API connections and reports status.
"""

import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()

def test_openai():
    """Test OpenAI connection."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return False, "No API key"

    try:
        import openai
        client = openai.OpenAI(api_key=key)
        start = time.perf_counter()
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=100,
            messages=[{"role": "user", "content": "Say 'OpenAI connection successful' in exactly those words."}]
        )
        latency = (time.perf_counter() - start) * 1000
        text = response.choices[0].message.content
        return True, f"OK ({latency:.0f}ms) - {text[:50]}"
    except Exception as e:
        return False, str(e)[:100]

def test_anthropic():
    """Test Anthropic/Claude connection."""
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return False, "No API key - set ANTHROPIC_API_KEY in .env"

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        start = time.perf_counter()
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            messages=[{"role": "user", "content": "Say 'Claude connection successful' in exactly those words."}]
        )
        latency = (time.perf_counter() - start) * 1000
        text = response.content[0].text
        return True, f"OK ({latency:.0f}ms) - {text[:50]}"
    except Exception as e:
        return False, str(e)[:100]

def test_gemini():
    """Test Google Gemini connection."""
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        return False, "No API key - set GEMINI_API_KEY in .env"

    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        start = time.perf_counter()
        response = model.generate_content("Say 'Gemini connection successful' in exactly those words.")
        latency = (time.perf_counter() - start) * 1000
        text = response.text
        return True, f"OK ({latency:.0f}ms) - {text[:50]}"
    except Exception as e:
        return False, str(e)[:100]


def main():
    print("=" * 60)
    print("PARALLAX PIPELINE - DIAGNOSTIC TEST")
    print("=" * 60)
    print()

    results = {}

    print("[1/3] Testing OpenAI (GPT-4o)...")
    ok, msg = test_openai()
    results["openai"] = ok
    print(f"      {'OK' if ok else 'FAILED'}: {msg}")
    print()

    print("[2/3] Testing Anthropic (Claude)...")
    ok, msg = test_anthropic()
    results["anthropic"] = ok
    print(f"      {'OK' if ok else 'FAILED'}: {msg}")
    print()

    print("[3/3] Testing Google (Gemini)...")
    ok, msg = test_gemini()
    results["gemini"] = ok
    print(f"      {'OK' if ok else 'FAILED'}: {msg}")
    print()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    working = sum(1 for v in results.values() if v)
    print(f"APIs available: {working}/3")
    print()

    if working == 3:
        print("All APIs configured! Run the full pipeline:")
        print("  python parallax_pipeline.py --test")
    else:
        print("Missing API keys. Add to .env file:")
        if not results["anthropic"]:
            print("  ANTHROPIC_API_KEY=sk-ant-...")
            print("  Get from: https://console.anthropic.com/")
        if not results["gemini"]:
            print("  GEMINI_API_KEY=AI...")
            print("  Get from: https://aistudio.google.com/apikey")
        print()

        if working >= 1:
            print("Partial test possible with available APIs.")
            print("Run: python test_single_api.py")

if __name__ == "__main__":
    main()
