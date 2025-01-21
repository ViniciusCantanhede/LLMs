import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import torch
from langchain_huggingface import ChatHuggingFace
from langchain_community.llms import HuggingFaceHub
from dotenv import load_dotenv
import os

load_dotenv()

# Acessar o token da OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")

# Verificar se o token foi carregado corretamente
if openai_api_key is None:
    print("Erro: O token da OpenAI não foi encontrado!")
else:
    print("Token da OpenAI carregado com sucesso.")

# Configurações do Streamlit
st.set_page_config(page_title="Seu assistente virtual 🤖", page_icon="🤖")
st.title("Seu assistente virtual 🤖")
#st.button("Botão")
#st.chat_input("Digite algo")

model_class = "openai"
#carregando modelo
def model_openai(model = "gpt-4o-mini", temperature = 0.1):
    llm = ChatOpenAI(model = model, api_key=openai_api_key, model_kwargs = {"temperature": temperature})
    return llm

def model_response(user_query, chat_history, model_class):
    if model_class == "openai":
        llm = model_openai()

    #definindo pormpt
    system_prompt = """
    Você é um assistente prestativo e está respondendo perguntas gerais. Responda em {language}.
    """
    language = "português"

    #adequando pipeline
    if model_class.startswith("hf"):
        user_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    else:
        user_prompt = "{input}"

    #criando prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", user_prompt)
    ])
    #criando chain
    chain = prompt_template | llm | StrOutputParser()
    #resposta
    return chain.stream({
        "chat_history": chat_history,
        "input": user_query,
        "language": language
    })
#configurando sessão
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content = "Olá! Como posso te ajudar?")]
#historico
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

user_query = st.chat_input("Digite sua mensagem...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content = user_query))
    with st.chat_message("Human"):
        st.markdown(user_query)
    with st.chat_message("AI"):
        resp = st.write_stream(model_response(user_query, st.session_state.chat_history, model_class))
    st.session_state.chat_history.append(AIMessage(content = resp))