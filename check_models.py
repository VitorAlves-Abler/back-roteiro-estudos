import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv() # Carrega o .env

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("Erro: GEMINI_API_KEY não configurada no .env")
else:
    genai.configure(api_key=API_KEY)
    print("Modelos disponíveis para sua API Key:")
    for m in genai.list_models():
        # Filtrar modelos que podem gerar conteúdo
        if 'generateContent' in m.supported_generation_methods:
            print(f"  - {m.name} (versão: {m.version})")