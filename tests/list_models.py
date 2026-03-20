"""
List available Gemini models for your API key.

Usage:
    python -m tests.list_models
"""
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY", ""))

for m in client.models.list():
    if "gemini" in m.name.lower():
        print(m.name)
