# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import google.generativeai as genai

# --- NOVO: Variável global para armazenar a instância do modelo, para cache ---
_cached_model = None

# Carrega as variáveis de ambiente
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("A variável de ambiente GEMINI_API_KEY não está configurada.")

# Configura o genai globalmente
genai.configure(api_key=API_KEY)

app = FastAPI()

class StudyTopic(BaseModel):
    topic: str

class SimpleRoteiroOutput(BaseModel):
    tema_solicitado: str
    roteiro: str

def get_model():
    """
    Função de inicialização preguiçosa (lazy initialization) do modelo.
    Inicializa o modelo apenas na primeira chamada e o reutiliza nas subsequentes.
    Isso é seguro para ambientes serverless.
    """
    global _cached_model
    if _cached_model is None:
        _cached_model = genai.GenerativeModel('gemini-2.5-flash')
    return _cached_model

@app.get("/")
async def read_root():
    return {"message": "Bem-vindo ao gerador de roteiros de estudo com IA!"}

@app.post("/gerar_roteiro/", response_model=SimpleRoteiroOutput)
async def gerar_roteiro(input_data: StudyTopic):
    topic = input_data.topic

    try:
        # Usa a função para obter o modelo, garantindo que ele seja inicializado corretamente
        model = get_model()
    except Exception as e:
        print(f"Erro ao inicializar o modelo da IA: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"O modelo de IA não pode ser inicializado. Erro: {e}"
        )

    prompt = f"""
    Crie um roteiro de estudo abrangente e detalhado para o seguinte tema: "{topic}".
    O roteiro deve incluir seções como introdução, fundamentos, tópicos avançados (se aplicável),
    e recursos adicionais. Apresente o roteiro de forma clara e legível, em texto corrido,
    com parágrafos e marcadores quando apropriado.
    Não inclua nenhuma saudação, despedida ou texto extra, apenas o roteiro.
    """

    generated_script = ""

    try:
        response = await model.generate_content_async(prompt)
        generated_script = response.text
    
    except Exception as e:
        print(f"Erro ao chamar a API do Gemini: {e}")
        error_detail = str(e) if str(e) else "Erro desconhecido da API de IA."
        raise HTTPException(
            status_code=500,
            detail=f"Não foi possível gerar o roteiro para '{topic}'. Erro: {error_detail}"
        )

    return {"tema_solicitado": topic, "roteiro": generated_script}