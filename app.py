from pymilvus import utility
from css import *
from functions import *

# Constants
BASE_URL = "https://docs.nvidia.com/cuda/"
DEPTH_LIMIT = 2
COLLECTION_NAME = "cuda_docs"

# Streamlit setup
st.markdown(page_bg_img, unsafe_allow_html=True)

# Main function
def main():
    st.title("Documentation Question Answering")

    # User input for website to crawl
    website_url = st.text_input("Enter the website URL to get data:", value=BASE_URL)

    if st.button("Get Data"):
        if website_url:
            with st.spinner('Getting and preparing data...'):
                st.session_state.data = crawl_website(website_url, DEPTH_LIMIT=DEPTH_LIMIT)
                st.session_state.chunks = chunk_data(st.session_state.data)
                st.session_state.word2vec_model = create_word2vec_model(st.session_state.chunks)
            
            with st.spinner('Setting up Milvus...'):
                connect_to_milvus_server()
                
                if utility.has_collection("cuda_docs"):
                    st.session_state.collection = Collection("cuda_docs")
                else:
                    st.session_state.collection = create_collection("cuda_docs")
                    create_index(st.session_state.collection)
                
                insert_data(st.session_state.collection, st.session_state.chunks, st.session_state.word2vec_model)
                st.session_state.collection.load()
            
            st.success("Crawling and data preparation completed!")
        else:
            st.warning("Please enter a website URL to crawl.")

    if 'collection' in st.session_state:
        st.markdown("### Ask a question about the website data")
        query = st.text_input("Enter your question:")
        if st.button("Get Answer"):
            if query:
                with st.spinner('Searching for answer...'):
                    answer = answer_query(query, st.session_state.collection, st.session_state.word2vec_model)
                st.markdown("<div class='answer-card'><h3>Answer</h3>" + answer + "</div>", unsafe_allow_html=True)
            else:
                st.warning("Please enter a question.")
    else:
        st.info("Please start crawling a website before asking questions.")

if __name__ == "__main__":
    main()