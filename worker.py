import os
import torch
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceHub
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models import Model

from langchain import PromptTemplate
#from langchain.chains import LLMChain, SimpleSequentialChain

# Verificar la disponibilidad de GPU y establecer el dispositivo apropiado para el cálculo.
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Variables globales
conversation_retrieval_chain = None
chat_history = []
llm_hub = None
embeddings = None

# Función para inicializar el modelo de lenguaje y sus embeddings
def init_llm():
    global llm_hub, embeddings

    my_credentials = {
        "url"    : "https://us-south.ml.cloud.ibm.com"
    }


    params = {
            GenParams.MAX_NEW_TOKENS: 256, # El número máximo de tokens que el modelo puede generar en una sola ejecución.
            GenParams.TEMPERATURE: 0.1,   # Un parámetro que controla la aleatoriedad de la generación de tokens. Un valor más bajo hace que la generación sea más determinista, mientras que un valor más alto introduce más aleatoriedad.
        }


    LLAMA2_model = Model(
            model_id= 'meta-llama/llama-3-8b-instruct', 
            credentials=my_credentials,
            params=params,
            project_id="skills-network"
            )



    llm_hub = WatsonxLLM(model=LLAMA2_model)

    ### --> si estás utilizando la API de huggingFace:
    # Configura la variable de entorno para HuggingFace e inicializa el modelo deseado, y carga el modelo en el HuggingFaceHub
    # no olvides eliminar llm_hub para watsonX

    # os.environ["HUGGINGFACEHUB_API_TOKEN"] = "TU CLAVE API"
    # model_id = "tiiuae/falcon-7b-instruct"
    #llm_hub = HuggingFaceHub(repo_id=model_id, model_kwargs={"temperature": 0.1, "max_new_tokens": 600, "max_length": 600})

    #Inicializa embeddings utilizando un modelo preentrenado para representar los datos de texto.
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": DEVICE}
    )


# Función para procesar un documento PDF
def process_document(document_path):
    global conversation_retrieval_chain

    # Cargar el documento
    loader = PyPDFLoader(document_path)
    documents = loader.load()

    # Dividir el documento en fragmentos
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    texts = text_splitter.split_documents(documents)

    # Crear una base de datos de embeddings utilizando Chroma a partir de los fragmentos de texto divididos.
    db = Chroma.from_documents(texts, embedding=embeddings)


    # --> Construir la cadena de QA, que utiliza el LLM y el recuperador para responder preguntas. 
    # Por defecto, el recuperador de vectorstore utiliza búsqueda por similitud. 
    # Si el vectorstore subyacente admite búsqueda de relevancia marginal máxima, puedes especificar eso como el tipo de búsqueda (search_type="mmr").
    # También puedes especificar argumentos de búsqueda como k para usar al hacer la recuperación. k representa cuántos resultados de búsqueda se envían al llm
    conversation_retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm_hub,
        chain_type="stuff",
        retriever=db.as_retriever(search_type="mmr", search_kwargs={'k': 6, 'lambda_mult': 0.25}),
        return_source_documents=False,
        input_key = "question"
     #   chain_type_kwargs={"prompt": prompt} # si estás utilizando una plantilla de prompt, necesitas descomentar esta parte
    )


# Función para procesar un aviso de usuario
def process_prompt(prompt):
    global conversation_retrieval_chain
    global chat_history

    # Consultar el modelo
    output = conversation_retrieval_chain({"question": prompt, "chat_history": chat_history})
    answer = output["result"]

    # Actualizar el historial de chat
    chat_history.append((prompt, answer))

    # Devolver la respuesta del modelo
    return answer

# Inicializar el modelo de lenguaje
init_llm()
