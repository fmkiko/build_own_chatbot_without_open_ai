import os
import torch
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Check for GPU availability and set the appropriate device for computation.
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Global variables
conversation_retrieval_chain = None
chat_history = []
llm_hub = None
embeddings = None

# Function to initialize the language model and its embeddings
def init_llm():
    global llm_hub, embeddings
    # Set up the environment variable for HuggingFace and initialize the desired model.
    # check token $ curl -X POST https://api-inference.huggingface.co/models/EleutherAI/gpt-neo-1.3B -H "Authorization: Bearer  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" -d '{"inputs":"Hola, mundo"}'
    #curl -X POST https://api-inference.huggingface.co/models/distilgpt2 -H "Authorization: Bearer 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" -H "Content-Type: application/json" do-d '{"inputs":"Hello, world"}'
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.environ["HUGGINGFACEHUB_API_TOKEN"]

    # repo name for the model
    # model_id = "tiiuae/falcon-7b-instruct"
    model_id = "distilgpt2" #" "meta-llama/Llama-2-7b-chat-hf
    # load the model into the HuggingFaceHub
    llm_hub = HuggingFaceHub(
        repo_id=model_id,
        huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
        model_kwargs={"temperature": 0.7, "max_new_tokens": 500}
    )

    # Initialize embeddings using a pre-trained model to represent the text data.
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": DEVICE}
    )

# Function to process a PDF document
def process_document(document_path):
    global conversation_retrieval_chain

    # Load the document
    loader = PyPDFLoader(document_path)
    documents = loader.load()
    
    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    texts = text_splitter.split_documents(documents)
    
    # Create an embeddings database using Chroma from the split text chunks.
    db = Chroma.from_documents(texts, embedding=embeddings)

    # Build the QA chain, which utilizes the LLM and retriever for answering questions.
    conversation_retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm_hub,
        chain_type="stuff",
        retriever=db.as_retriever(search_type="mmr", search_kwargs={'k': 6, 'lambda_mult': 0.25}),
        return_source_documents=False,
        input_key="question"
    )

# Function to process a user prompt
def process_prompt(prompt):
    global conversation_retrieval_chain
    global chat_history
    
    # Query the model
    output = conversation_retrieval_chain({"question": prompt, "chat_history": chat_history})
    answer = output["result"]
    
    # Update the chat history
    chat_history.append((prompt, answer))
    
    # Return the model's response
    return answer

# Initialize the language model
init_llm()