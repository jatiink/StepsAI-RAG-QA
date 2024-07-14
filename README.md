# Documentation Question Answering System

This project implements a question answering system for documentation using web crawling, vector embeddings, and the Milvus vector database.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Docker Setup for Milvus](#docker-setup-for-milvus)

## Prerequisites

- Python 3.7+
- Docker
- Milvus 2.0+

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/documentation-qa-system.git
cd documentation-qa-system
```

2. Install the required Python packages:
```
pip install -r requirements.txt
```

3. Set up Milvus using Docker (see Docker Setup for Milvus section below).
## Usage
1. Start the Streamlit app:
```
streamlit run app.py
```
2. Open your web browser and navigate to the URL provided by Streamlit (usually http://localhost:8501).
3. Enter the URL of the documentation website you want to crawl and analyze.
4. Click "Get Data" to start the crawling and data preparation process.
5. Once the data is prepared, you can ask questions about the documentation in the text input field

## File Structure
* app.py: Main Streamlit application file
* css.py: CSS styles for the Streamlit app
* functions.py: Helper functions for web crawling, data processing, and Milvus operations
* requirements.txt: List of required Python packages

## Docker Setup for Milvus
To run Milvus using Docker, follow these steps:
*  Download the Milvus Docker Compose file:
```
wget https://github.com/milvus-io/milvus/releases/download/v2.0.2/milvus-standalone-docker-compose.yml -O docker-compose.yml
```
*  Start Milvus:
```
docker-compose up -d
```
*  Verify that Milvus is running:
```
docker-compose ps
```
*  You should see three containers running: milvus-standalone, milvus-etcd, and milvus-minio.
*  To stop Milvus:
```
docker-compose down
```
*  To delete Milvus data after stopping:
```
rm -rf volumes
```

* For more detailed information on Milvus setup and configuration, please refer to the [official Milvus documentation](https://milvus.io/docs).
## Key Components
### Web Crawling
* The system uses requests and BeautifulSoup to crawl the CUDA documentation website. The crawl_website function in functions.py handles the crawling process with a depth limit.
### Text Processing
* The crawled text is preprocessed using Gensim's simple_preprocess function.
* The preprocessed text is then chunked using LangChain's RecursiveCharacterTextSplitter.
### Embedding Generation
* A Word2Vec model is created using the preprocessed text chunks.
* Document embeddings are generated by averaging the word vectors for each chunk.
### Vector Database
* Milvus is used as the vector database for storing and searching document embeddings.
* The system creates a collection named "cuda_docs" with fields for id, text, and embedding.
### Question Answering
* The system uses hybrid retrieval to find relevant documents based on the user's query.
* An Ollama LLM (llama3 model) is used to generate answers based on the retrieved context and the user's question.
### User Interface
* The user interface is built using Streamlit, providing an easy-to-use web application for interacting with the question answering system.
### Notes
* Ensure that Milvus is running before starting the Streamlit app.
* The system is designed to work with CUDA documentation but can be adapted for other documentation websites by modifying the BASE_URL in app.py.
* For optimal performance, consider using a CUDA-compatible GPU for faster processing of embeddings and LLM inference.
