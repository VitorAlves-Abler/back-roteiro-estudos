# main.py
from fastapi import FastAPI, HTTPException # <<< ADICIONADO: HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import google.generativeai as genai
import asyncio # <<< ADICIONADO: para usar asyncio.wait_for para timeout

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Pega a chave de API da variável de ambiente
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("A variável de ambiente GEMINI_API_KEY não está configurada.")

# Configura a biblioteca do Google AI com sua chave
genai.configure(api_key=API_KEY)

# Inicializa o modelo Gemini
model = genai.GenerativeModel('gemini-2.5-flash')

app = FastAPI()

# Define um modelo Pydantic para o input do tema de estudo
class StudyTopic(BaseModel): # <<< RENOMEADO: studyTopic para StudyTopic (convenção de classes)
    topic: str

# <<< NOVO: Modelo Pydantic para o output simples do roteiro (agora uma string)
class SimpleRoteiroOutput(BaseModel):
    tema_solicitado: str
    roteiro: str

# Endpoint de teste básico
@app.get("/")
async def read_root():
    return {"message": "Bem-vindo ao gerador de roteiros de estudo com IA!"}

# NOVO ENDPOINT: Gerar Roteiro
@app.post("/gerar_roteiro/", response_model=SimpleRoteiroOutput) # <<< ADICIONADO: response_model
async def gerar_roteiro(input_data: StudyTopic): # <<< RENOMEADO: studyTopic para StudyTopic
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

    generated_script = "" # <<< RENOMEADO: generatedScript para generated_script (convenção python)

    try:
        # Define um tempo limite para a requisição à API do Gemini (ex: 30 segundos)
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
        # Retorna uma mensagem de erro clara para o frontend
        raise HTTPException(
            status_code=500, # 500 Internal Server Error
            detail=f"Não foi possível gerar o roteiro para '{topic}' devido a um erro no serviço de IA. Detalhes: {e}"
        )

    return {"tema_solicitado": topic, "roteiro": generated_script}