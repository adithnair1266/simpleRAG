from dotenv import load_dotenv
from fastapi import FastAPI, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Annotated
import os
import regex as re
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from langchain.memory import ConversationSummaryMemory
from langchain_community.document_loaders import  PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama





#Function to get extension of the uploaded file
def get_file_extension(filename):
    
    match = re.search(r'\.([^.]+)$', filename)
    if match:
        return match.group(1)
    else:
        return None

#Function to create a vector database 
def createDB(file, embeddings):
    extension = get_file_extension(file.filename)
    if extension == 'txt':
            loader = TextLoader(f'uploads/{file.filename}')

    elif extension == 'doc' or extension == "docx":
            loader = Docx2txtLoader(f'uploads/{file.filename}')

    elif extension == 'pdf':
            loader = PyMuPDFLoader(f'uploads/{file.filename}')

    else:
        print("FileType Not supported")

        
    docs = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=75)
    documents = text_splitter.split_documents(docs)    

        
    db = FAISS.from_documents(documents,embeddings)
    db.save_local(f"D:/project_RAG/llm-assignment-master/backend/vectordb/faiss-index-{file.filename}")


#Return the similar context and prompt the local model to return related answer
def getQuery(file, question,  embeddings ):
   
    new_DB = FAISS.load_local(f"D:/project_RAG/llm-assignment-master/backend/vectordb/faiss-index-{file.filename}", embeddings, allow_dangerous_deserialization=True)

    prompt = ChatPromptTemplate.from_template(
            """
             You are an assistant for question answering tasks. Use the following context to answer the question in one line.
            If you don't know the answer, just say you don't know. DO NOT ADD NON RELEVANT INFORMATION.
            Question: {question} 
            Context: {context} 
            """
        )





    rag = RetrievalQA.from_chain_type(
                llm=ChatOllama(model="dolphin-phi"),
                retriever=new_DB.as_retriever(),
                memory=ConversationSummaryMemory(llm = ChatOllama(model="dolphin-phi")),
                chain_type_kwargs={"prompt": prompt, "verbose": True},
            )

    answer = (rag.invoke(question)['result'])
    return answer



load_dotenv()



class Response(BaseModel):
    result: str | None


origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000",
    "http://localhost:3001",
    "http://100.118.57.8:3000"
]


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.post("/predict", response_model = Response)
async def predict(question: str = Form(...), file: UploadFile = File(...)):

    #main function
    modelPath = "sentence-transformers/all-MiniLM-l6-v2"
    model_kwargs = {'device':'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
            model_name=modelPath,     
            model_kwargs=model_kwargs, 
            encode_kwargs=encode_kwargs

        )

    with open(f"uploads/{file.filename}" , 'wb') as f:
            f.write(await file.read()) 
    
    createDB(file, embeddings)
    answer = getQuery(file, question, embeddings)

    return {"result": answer}









