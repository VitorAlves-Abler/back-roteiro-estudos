# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Carrega as variáveis de ambiente do arquivo .env
# Isso é seguro no Vercel pois as variáveis de ambiente são injetadas
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    # Levanta o erro se a chave não estiver configurada no Vercel
    raise ValueError("A variável de ambiente GEMINI_API_KEY não está configurada.")

# --- NOVO: Configura o genai globalmente, mas não inicializa o modelo ---
genai.configure(api_key=API_KEY)

# Cria a instância da aplicação FastAPI
app = FastAPI()

# Modelos Pydantic (permanecem os mesmos)
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
    topic = input_data.topic

    # --- NOVO: Inicialização do modelo dentro da função (ou de forma lazy) ---
    try:
        # A Vercel reusa o processo, então esta linha será muito rápida após a primeira chamada
        model = genai.GenerativeModel('gemini-2.5-flash')
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