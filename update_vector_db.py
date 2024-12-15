import os
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import torch

def update_vector_database(input_files, output_db_path):
    """
    Create or update a vector database from multiple Wikipedia article text files.
    
    Args:
        input_files (list): List of paths to input text files
        output_db_path (str): Directory to save the vector database
    """
    # 2. Text Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Moderate chunk size to capture meaningful context
        chunk_overlap=50,  # Small overlap to preserve context between chunks
        length_function=len,
        is_separator_regex=False
    )
    
    # 3. Embedding Generation
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )
    
    # Check if an existing vector database exists
    vectorstore = None
    try:
        # Try to load existing vector database with dangerous deserialization allowed
        vectorstore = FAISS.load_local(
            output_db_path, 
            embedding_model, 
            allow_dangerous_deserialization=True
        )
        print("Existing vector database loaded successfully.")
    except Exception as e:
        print(f"Could not load existing database: {e}")
    
    # Process each input file
    total_chunks = 0
    for input_file in input_files:
        # 1. Read the scraped text
        with open(input_file, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Split the text into chunks
        chunks = text_splitter.split_text(text)
        total_chunks += len(chunks)
        
        # Convert chunks to Document objects
        documents = [
            Document(page_content=chunk, metadata={"source": input_file}) 
            for chunk in chunks
        ]
        
        # Create or update vector store
        if vectorstore is None:
            # Create new vector store if it doesn't exist
            vectorstore = FAISS.from_documents(
                documents, 
                embedding_model
            )
            print(f"Created new vector database with chunks from {input_file}")
        else:
            # Generate embeddings for new chunks
            new_embeddings = embedding_model.embed_documents(chunks)
            
            # Convert to numpy array
            new_embeddings_array = np.array(new_embeddings)
            
            # Add new documents to the store
            vectorstore.add_documents(documents)
            
            # Extend the existing index with new embeddings
            vectorstore.index.add(new_embeddings_array)
            
            print(f"Added chunks from {input_file} to existing vector database")
    
    # 5. Export Updated Vector Database
    # Create output directory if it doesn't exist
    os.makedirs(output_db_path, exist_ok=True)
    
    # Save vector database
    vectorstore.save_local(output_db_path)
    
    print(f"Vector database updated and saved to {output_db_path}")
    print(f"Total chunks in database: {vectorstore.index.ntotal}")
    
    return vectorstore

# Example usage
if __name__ == "__main__":
    # Specify your input files
    input_files = [
        "bhabha.txt",
        "raman.txt",
    ]
    
    output_path = "./wikipedia_vector_db"  # Output directory for vector database
    
    vector_db = update_vector_database(input_files, output_path)
    
    # Optional: Demonstrate retrieval
    query = "What are the key topics in these articles?"
    retrieved_docs = vector_db.similarity_search(query, k=3)
    
    print("\nRetrieved Chunks:")
    for doc in retrieved_docs:
        print(doc.page_content)