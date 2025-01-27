import os
import uvicorn
from io import BytesIO
from dotenv import load_dotenv
from fastapi import FastAPI, status, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Dict, List
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import Blob
from langchain_community.document_loaders.parsers import PyMuPDFParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

class QueryModel(BaseModel):
    """
    Pydantic class to validate user's query
    """

    user_query: str

# Load enviornment variables
load_dotenv()
# Variables to be used to instantiate models
MODEL_API_KEY = os.environ.get('API_KEY')
CHAT_MODEL_NAME = os.environ.get('CHAT_MODEL_NAME')
EMBEDDING_MODEL_NAME = os.environ.get('EMBEDDING_MODEL_NAME')
SYSTEM_MESSAGE = os.environ.get('SYSTEM_MESSAGE')
CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE'))
CHUNK_OVERLAP = int(os.environ.get('CHUNK_OVERLAP'))
CACHE_DIRECTORY = os.environ.get('CACHE_DIRECTORY')
K_RETRIEVALS = int(os.environ.get('K_RETRIEVALS'))
COLLECTION_NAME = os.environ.get('COLLECTION_NAME')

# Variables to be used by embedding model and vector store
EMBEDDING_MODEL_KWARGS = {"device": "cpu"}
EMBEDDING_ENCODE_KWARGS = {'normalize_embeddings': False}
EMBEDDING_MODEL_CACHE_DIRECTORY = os.path.join(os.getcwd(), 'EMBEDDING_MODELS')
VECTOR_STORE_PERSIST_DIRECTORY = os.path.join(os.getcwd(), 'VECTOR_STORE_DIR')

# Global variables
chat_model = ChatOpenAI(api_key=MODEL_API_KEY,
                        model=CHAT_MODEL_NAME,
                        temperature=0.1,
                        max_retries=5,
                        streaming=True)
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME,
                                        model_kwargs=EMBEDDING_MODEL_KWARGS,
                                        encode_kwargs=EMBEDDING_ENCODE_KWARGS,
                                        cache_folder=EMBEDDING_MODEL_CACHE_DIRECTORY)
vector_store = Chroma(collection_name='mango_projects',
                      persist_directory=VECTOR_STORE_PERSIST_DIRECTORY,
                      embedding_function=embedding_model)
pdf_parser = PyMuPDFParser(mode='single')
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500)
chat_prompt_template = ChatPromptTemplate(messages=[SystemMessage(content=SYSTEM_MESSAGE),
                                                    HumanMessagePromptTemplate(
                                                        prompt=PromptTemplate(input_variables=['retrieved_docs', 'user_query'],
                                                                              template='User Query:\n{user_query}\n\nRetrieved Documents:\n{retrieved_docs}'))])
# Instantiate FastAPI
app = FastAPI()

@app.post('/chat', status_code=200)
async def get_answer(request_model: QueryModel):
    """
    Path operation to generate ai's response based on retrieved information and user's query.

    :Parameters:
    requset_model: User's query to be processed by the chat model.

    :Returns:
    Response streamed using the user's query and retrieved documents
    """
    # Load global variables 
    global vector_store, K_RETRIEVALS, chat_model, chat_prompt_template

    # Retrieve documents from the vector store
    retrieved_documents: List[Document] = await vector_store.asimilarity_search(query=request_model.user_query,
                                                                                k=K_RETRIEVALS)
    # Store all retrieved documents as a single string
    retrieved_document_str = '\n\n'.join([doc.page_content for doc in retrieved_documents])

    # Stream chains output 
    llm_chain = chat_prompt_template | chat_model | StrOutputParser()
    async def token_generator():
        async for token in llm_chain.astream({'user_query': request_model.user_query, 'retrieved_docs': retrieved_document_str}):
            yield token
    
    return StreamingResponse(token_generator(), media_type="text/plain")


@app.post('/upsert', status_code=status.HTTP_201_CREATED)
async def load_documents(upload_file: UploadFile):
    """
    Path operation to load documents into the database"
    
    :Parameters:
    files: List of files to be uploaded

    :Returns:
    A response message to indicate the status of the operation
    """
    # Load global variables for document storing
    global vector_store, text_splitter

    file_content_blob = Blob.from_data(BytesIO(upload_file.file.read()).getvalue())
    parsed_file_content = PyMuPDFParser(mode='single').parse(blob=file_content_blob)
    documents: List[Document] = text_splitter.split_documents(documents=parsed_file_content)
    for document in documents:
        document.metadata['source'] = upload_file.filename

    await vector_store.aadd_documents(documents=documents)

    return JSONResponse(jsonable_encoder({"message": "File has been uploaded successfully"}))

if __name__ == '__main__':
    uvicorn.run(app=app, host='127.0.0.1', port=3000)