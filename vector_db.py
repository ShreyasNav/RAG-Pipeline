import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import torch
from typing import List

def create_vector_database(input_file_path: str, output_db_path: str):
    """
    Create a vector database from a Wikipedia article text file.
    
    Args:
        input_file_path (str): Path to the input text file
        output_db_path (str): Directory to save the vector database
    
    Returns:
        FAISS: Created vector database
    """
    # 1. Read the scraped text
    with open(input_file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # 2. Text Chunking
    # Rationale for chunking strategy:
    # - Use RecursiveCharacterTextSplitter for semantic-aware splitting
    # - Chunk size of 500 characters balances context preservation and computational efficiency
    # - 50 character overlap helps maintain context across chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  
        chunk_overlap=50,  
        length_function=len,
        is_separator_regex=False
    )
    
    # Split the text into chunks
    chunks: List[str] = text_splitter.split_text(text)
    
    # 3. Embedding Generation
    # Using all-MiniLM-L6-v2 model:
    # - Lightweight 
    # - Good balance of performance and computational efficiency
    # - Supports multiple languages
    # - Generates 384-dimensional embeddings
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )
    
    # 4. Vector Database Creation
    # Using FAISS (Facebook AI Similarity Search):
    # Benefits:
    # - Extremely fast similarity search
    # - Efficient memory usage
    # - Supports GPU acceleration
    # - Open-source
    # Drawbacks:
    # - Requires manual index management
    # - Less flexible than some cloud-based solutions
    # - No built-in persistence (we'll use FAISS's save/load methods)

    # Generate embeddings explicitly to ensure 1:1 mapping
    embeddings = embedding_model.embed_documents(chunks)
    
    text_embedding_pairs: List[Tuple[str, List[float]]] = list(zip(chunks, embeddings))

    # Create vector store with explicit embeddings
    vectorstore = FAISS.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=embedding_model
    )
    
    # 5. Export Vector Database
    # Create output directory if it doesn't exist
    os.makedirs(output_db_path, exist_ok=True)
    
    # Save vector database
    vectorstore.save_local(output_db_path)
    
    print(f"Vector database created and saved to {output_db_path}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Total embeddings: {len(embeddings)}")

    return vectorstore

# Example usage
if __name__ == "__main__":
    input_file = "text.txt"
    output_path = "./wikipedia2_vector_db"  # Output directory for vector database
    
    vector_db = create_vector_database(input_file, output_path)
    
    # Demonstrate retrieval
    query = "What is the main topic of the article?"
    retrieved_docs = vector_db.similarity_search(query, k=3)
    
    print("\nRetrieved Chunks:")
    for doc in retrieved_docs:
        print(doc.page_content)