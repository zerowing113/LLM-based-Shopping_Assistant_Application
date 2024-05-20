import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env")
GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY", "input GEMINI")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "input PINECONE_API_KEY")
OPENAI = os.environ.get("OPENAI", "input OPENAI")