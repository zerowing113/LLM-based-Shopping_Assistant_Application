from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from config import GOOGLE_API_KEY
import os
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
from langchain_google_genai import GoogleGenerativeAI
from llama_index.llms.langchain import LangChainLLM


llamaindex_llm = LangChainLLM(llm=GoogleGenerativeAI(model="gemini-pro"))
# define embedding function
llamaindex_embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


langchain_llm = GoogleGenerativeAI(model="gemini-pro")

print("LLM-based Shopping Assistant Application is ready to use!")