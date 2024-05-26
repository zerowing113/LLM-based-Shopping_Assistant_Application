import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env")
#GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY", "input GEMINI")
GOOGLE_API_KEY = "AIzaSyA4XNrms6DClzcgu6CSlwhYuSLYwUN2uv4"
#PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "input PINECONE_API_KEY")
PINECONE_API_KEY = "60b6bd20-fe0f-454e-88c4-4d1e28293dab"
OPENAI = os.environ.get("OPENAI", "input OPENAI")