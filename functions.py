import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
from langchain_community.llms import Ollama
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import time
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.models.phrases import Phrases, Phraser
from urllib.parse import urlparse
import streamlit as st

# Function to check if a URL is valid
def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

# Function to crawl the website
def crawl_website(url, depth=0, visited=None, DEPTH_LIMIT=0):
    if visited is None:
        visited = set()
    if depth > DEPTH_LIMIT or url in visited:
        return []
    visited.add(url)
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    links = [urljoin(url, a['href']) for a in soup.find_all('a', href=True)]
    data = [soup.get_text()]
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(crawl_website, link, depth + 1, visited) 
                   for link in links if re.match(r'https://docs\.nvidia\.com/cuda/.*', link)]
        for future in futures:
            data.extend(future.result())
    return data

# Preprocess text using Gensim's simple_preprocess
def preprocess_text(text):
    return simple_preprocess(text, deacc=True)

# Create Word2Vec model
def create_word2vec_model(chunks, VECTOR_DIM=768):
    sentences = [preprocess_text(chunk) for chunk in chunks]
    phrases = Phrases(sentences, min_count=30, progress_per=10000)
    bigram = Phraser(phrases)
    sentences = bigram[sentences]
    model = Word2Vec(sentences, vector_size=VECTOR_DIM, window=5, min_count=1, workers=4)
    return model

# Get document embedding using Word2Vec
def get_doc_embedding(doc, model, VECTOR_DIM=768):
    words = preprocess_text(doc)
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if not word_vectors:
        return np.zeros(VECTOR_DIM)
    return np.mean(word_vectors, axis=0)

# Chunk data (modified to use Gensim's preprocessing)
def chunk_data(scraped_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(" ".join(scraped_data))
    return [" ".join(preprocess_text(chunk)) for chunk in chunks]

# Connect to Milvus server
def connect_to_milvus_server():
    try:
        connections.connect(
            alias="default", 
            host="localhost", 
            port="19530",
            timeout=60
        )
        st.success("Connected to Milvus server successfully!")
        return True
    except Exception as e:
        st.error(f"Failed to connect to Milvus server: {str(e)}")
        return False

# Create collection
def create_collection(collection_name, dim=768):
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    schema = CollectionSchema(fields, "CUDA documentation collection")
    collection = Collection(collection_name, schema)
    st.markdown(f"Collection schema: {collection.schema}")
    return collection

# Create index
def create_index(collection, field_name="embedding"):
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024}
    }
    collection.create_index(field_name, index_params)
    st.success(f"Index created on field '{field_name}' successfully!")

# Insert data
def insert_data(collection, chunks, word2vec_model):
    try:
        start_time = time.time()
        embeddings = [get_doc_embedding(chunk, word2vec_model) for chunk in chunks]
        entities = [
            chunks,  # text
            embeddings  # embedding
        ]
        collection.insert(entities)
        collection.flush()
        end_time = time.time()
        st.success(f"Data inserted successfully in {end_time - start_time:.2f} seconds!")
    except Exception as e:
        st.error(f"Failed to insert data: {str(e)}")
        st.error(f"Chunks length: {len(chunks)}, Embeddings length: {len(embeddings)}")
        st.error(f"First chunk: {chunks[0][:100]}")  # Display first 100 characters of the first chunk
        st.error(f"First embedding shape: {embeddings[0].shape}")

# Hybrid retrieval
def hybrid_retrieval(query, collection, word2vec_model):
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    query_embedding = get_doc_embedding(query, word2vec_model)
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=5,
        output_fields=["text"]
    )
    return [hit.entity.get('text') for hit in results[0]]

# Answer query
def answer_query(query, collection, word2vec_model):
    retrieved_docs = hybrid_retrieval(query, collection, word2vec_model)
    context = " ".join(retrieved_docs)
    llm = Ollama(model="llama2")
    response = llm(f"Context: {context}\nQuestion: {query}")
    return response