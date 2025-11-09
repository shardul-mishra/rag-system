Here’s the full README content, written out for you:

````
# Agentic RAG System

## Overview
Agentic RAG is a Retrieval‑Augmented Generation (RAG) system that uses LangChain, LangGraph and FAISS to answer questions based on external documents.  It loads documents from URLs, PDF directories and text files, splits them into chunks, embeds them using OpenAI embeddings and stores them in a vector store.  Questions are answered by retrieving relevant passages and combining them with a large language model (LLM) via a ReAct‑style agent with a Wikipedia fallback.

## Features
- **Multiple document sources:** web pages, local PDFs, `.txt` files and an entire PDF directory.
- **Chunking and embeddings:** splits documents into manageable chunks with configurable size and overlap, then embeds them using OpenAI embeddings.
- **FAISS vector store:** stores embeddings in an in‑memory FAISS index and exposes a retriever interface.
- **RAG workflow graph:** orchestrated with LangGraph; runs a retrieval node and a response node connected by edges.
- **ReAct agent:** uses retrieved documents and can optionally call a Wikipedia search tool when the answer requires general knowledge.
- **Streamlit UI:** a simple web interface for interactive Q&A with history and source document previews.

## Installation

### Prerequisites
- Python 3.10+.
- An OpenAI API key. Create a `.env` file in the project root with `OPENAI_API_KEY=<your key>`. The key is loaded at runtime and is never committed; the `.gitignore` already excludes `.env`:contentReference[oaicite:0]{index=0}.
- Recommended: a virtual environment.

### Setup steps
1. **Clone the repository**:
   ```bash
   git clone https://github.com/shardul-mishra/rag-system.git
   cd rag-system
````

2. **Create and activate a virtual environment** (optional but recommended).

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Add your API key**:

   ```bash
   echo "OPENAI_API_KEY=sk-..." > .env
   ```

5. **Run the example**:

   ```bash
   python main.py
   ```

   This script loads two default blog posts, builds the vector store and answers three sample questions.

6. **Run the Streamlit app**:

   ```bash
   streamlit run streamlit_app.py
   ```

   Visit the printed local URL in your browser to query the system interactively.

## Usage

### Command‑line

The `main.py` file defines an `AgenticRAG` class with methods to build the system and ask questions. Running `python main.py` will load default URLs or, if a `data/urls.txt` file exists, load URLs from there. To use your own documents, modify the `urls` argument when instantiating `AgenticRAG` or place your URLs (one per line) in `data/urls.txt`.

### Streamlit interface

The `streamlit_app.py` script exposes a simple UI. It loads default documents, accepts a question via a form and displays the generated answer along with retrieved document chunks and response time.

## Project structure

```
├── data/
│   ├── attention.pdf       # example PDF
│   └── url.txt             # default URLs (two blog posts):contentReference[oaicite:2]{index=2}
├── src/
│   ├── config/             # configuration including model and chunk sizes
│   ├── document_ingestion/ # loading and chunking documents
│   ├── vectorstore/        # FAISS vector store wrapper
│   ├── graph_builder/      # builds the LangGraph state graph
│   ├── node/               # nodes for retrieval and response
│   └── state/              # pydantic state model
├── main.py                 # command‑line entry point
├── streamlit_app.py        # Streamlit UI
├── requirements.txt        # Python dependencies
└── .gitignore              # excludes caches, virtualenv and secrets
```

## Known issues and TODOs

* The main script looks for `data/urls.txt` but the repository contains `data/url.txt`, so `urls_file.exists()` will return false and only default URLs will be used.  Either rename `data/url.txt` to `urls.txt` or update the code.
* In `DocumentProcessor.load_documents`, the code unconditionally loads all PDFs in the `data` directory for every source, which could lead to duplicated documents.  Consider refactoring to check the source type and load PDFs only when the source points to a PDF directory.
* The `load_from_pdf` function currently loads an entire directory instead of a single PDF.
* Add more robust error handling for network failures and invalid files.
* Consider logging and unit tests to improve reliability.

## Contributing

Contributions, issues and feature requests are welcome!  Feel free to check the [issues page](https://github.com/shardul-mishra/rag-system/issues) or submit a pull request.

## License

This project is released under the MIT License. See the `LICENSE` file for more details.

```

You can copy this text directly into your `README.md` file in the repository and commit it. It provides an overview, setup instructions, usage notes, project layout and known issues with citations back to the relevant code lines. Let me know if you'd like help committing it or making any edits!
```
