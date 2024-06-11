# Welcome to simpleRAG

This is a simple web application that implements RAG (Retrieval-Augmentation-Generation) using a local LLM and the LangChain framework. Uses React for front end and fastAPI for backend serves. You can upload any text-containing file (currently supporting .txt, .doc, .docx files) and ask relevant questions. Try it out yourself!

## Project Structure

In this project, you will find two main directories:

1. `backend`: Contains the server-side **Python** code.
2. `frontend`: Contains the client-side **TypeScript** code.

## Backend

### Requirements

- Python 3.10
- Ollama models (local). Ensure you have Ollama installed.

### Setup

1. Install the necessary Python packages using:

    ```bash
    pip install -r requirements.txt
    ```

### Running the Backend Server

To launch the server, navigate to the `backend` directory and run:

```bash
uvicorn main:app --reload
```

This will start the server at [http://127.0.0.1:8000/](http://127.0.0.1:8000/).

## Frontend

The project structure within the `frontend` directory follows the official `create-react-app` structure as described in the [documentation](https://create-react-app.dev/docs/folder-structure).

### Requirements

- Node.js v20.11.1

### Setup

1. Navigate to the `frontend` directory and install the necessary packages:

    ```bash
    npm install
    ```

2. Launch the app in development mode:

    ```bash
    npm start
    ```

   This will launch the app. Open [http://localhost:3000](http://localhost:3000) to view it in your browser.

## What is RAG?

### Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) is a cutting-edge approach in natural language processing that blends the best of both worlds: retrieval-based and generation-based models. Imagine having an AI that not only pulls in relevant information from a massive database but also uses that information to craft well-informed and contextually rich responses or content.

#### How It Works

1. **Retrieval**: The RAG model searches through a large pool of documents to find pieces of information that are most relevant to your query.
2. **Generation**: It then takes this information and generates responses that are accurate and deeply contextual.

This makes RAG incredibly effective for tasks like answering questions, creating detailed articles, and summarizing complex topics.

---

We hope you find simpleRAG useful for your projects and enjoy exploring the capabilities of Retrieval-Augmented Generation. Happy exploring!
