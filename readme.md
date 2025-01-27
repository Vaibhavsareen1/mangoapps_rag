# RAG Frontend and Backend Application

This project is a Retrieval-Augmented Generation (RAG) application that consists of a frontend built with Gradio and a backend built with FastAPI. The application allows users to upload documents and interact with a language model to retrieve information from the uploaded documents.

## Project Structure

- `.env`: Environment variables required for the application.
- `backend_main.py`: Backend server implementation using FastAPI.
- `frontend_main.py`: Frontend implementation using Gradio.
- `requirements.txt`: List of dependencies required for the project.

## Setup

1. Clone the repository:
git clone <repository-url>
cd <repository-directory>

2. Create a virtual environment and activate it:
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

3. Install the dependencies:
pip install -r requirements.txt

4. Create a .env file in the root directory and add the following environment variables:

API_KEY="YOUR_API_KEY"
CHAT_MODEL_NAME="gpt-4o-mini"
EMBEDDING_MODEL_NAME="sentence-transformers/all-mpnet-base-v2"
SYSTEM_MESSAGE="You are an AI assistant specialized in answering user's query. Your task is to assist and answer user's query generating accurate and contextually appropriate responses. You are to only provide an answer based on retrieved information and not generate new information."
CHUNK_SIZE=500
CHUNK_OVERLAP=50
K_RETRIEVALS=5
COLLECTION_NAME="mangoapps_project"

## Running the Application
1. Start the backend server:
python backend_main.py

The backend server will be running at http://127.0.0.1:3000.

2. Start the frontend application:
python frontend_main.py

The frontend application will be running at http://127.0.0.1:8002.

## Usage
    1. Open the frontend application in your browser at http://127.0.0.1:8002.
    2. Upload PDF documents using the document upload interface.
    3. Interact with the language model by entering queries in the chat interface.