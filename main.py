# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import google.generativeai as genai
import asyncio

# Carrega as variáveis de ambiente do arquivo .env
if os.path.exists(".env"):
    load_dotenv()

# Pega a chave de API da variável de ambiente
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("A variável de ambiente GEMINI_API_KEY não está configurada.")

# Configura a biblioteca do Google AI
genai.configure(api_key=API_KEY)
# Inicializa o modelo Gemini

model = genai.GenerativeModel('gemini-2.5-flash')

app = FastAPI()

class StudyTopic(BaseModel):
    topic: str

class SimpleRoteiroOutput(BaseModel):
    tema_solicitado: str
    roteiro: str

@app.get("/")
async def read_root():
    return {"message": "Bem-vindo ao gerador de roteiros de estudo com IA!"}

@app.post("/gerar_roteiro/", response_model=SimpleRoteiroOutput)
async def gerar_roteiro(input_data: StudyTopic):
    """
    Endpoint que recebe um tema de estudo e retorna um roteiro gerado por IA.
    """
    topic = input_data.topic

    prompt = f"""
    Crie um roteiro de estudo abrangente e detalhado para o seguinte tema: "{topic}".
    O roteiro deve incluir seções como introdução, fundamentos, tópicos avançados (se aplicável),
    e recursos adicionais. Apresente o roteiro de forma clara e legível, em texto corrido,
    com parágrafos e marcadores quando apropriado.
    Não inclua nenhuma saudação, despedida ou texto extra, apenas o roteiro.
    """

    generated_script = ""
    
    try:
        # Define um tempo limite para a requisição à API do Gemini
        # Se a requisição demorar mais que isso, um TimeoutError será levantado
        response = await asyncio.wait_for(
            model.generate_content_async(prompt),
            timeout=60.0
        )
        # Acessa o texto gerado pela IA.
        generated_script = response.text
    
    except asyncio.TimeoutError: # <<< TRATAMENTO DE ERRO DE TIMEOUT
        print(f"TIMEOUT: A requisição ao Gemini demorou demais para '{topic}'.")
        raise HTTPException(
            status_code=504, # 504 Gateway Timeout
            detail=f"O serviço de IA demorou muito para responder para o tema '{topic}'. Por favor, tente novamente."
        )
    
    except Exception as e:
        print(f"Erro ao chamar a API do Gemini para '{topic}': {e}")
        # Retorna uma mensagem de erro
        raise HTTPException(
            status_code=500, # 500 Internal Server Error
            detail=f"Não foi possível gerar o roteiro para '{topic}' devido a um erro no serviço de IA. Detalhes: {e}"
        )
    
    return {"tema_solicitado": topic, "roteiro": generated_script}