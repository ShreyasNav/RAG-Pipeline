from fastapi import FastAPI, UploadFile, HTTPException
import os
import tempfile
import soundfile as sf
import torch
import requests
import faiss
import numpy as np
import json
import pickle
from sentence_transformers import SentenceTransformer

# Import the ASR model
import nemo.collections.asr as nemo_asr

# Load the NeMo ASR model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "ai4b_indicConformer_hi.nemo"

try:
    model = nemo_asr.models.EncDecCTCModel.restore_from(restore_path=MODEL_PATH)
    model.freeze()
    model = model.to(DEVICE)
    model.cur_decoder = "ctc"
except Exception as e:
    raise RuntimeError(f"Error loading ASR model: {e}")

# Load sentence transformer for question embedding
sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Load vector database
VECTOR_DB_PATH = "wikipedia2_vector_db/index.faiss"
index = faiss.read_index(VECTOR_DB_PATH)

pkl_path = "wikipedia2_vector_db/index.pkl"

# Cohere API Key and model
COHERE_API_KEY = "Rpcu5vm5NUmrBpD1PhiIJAL4EAOVXreRUmggd0Wr"
COHERE_MODEL = "command-r-08-2024"

# Initialize FastAPI app
app = FastAPI()

def transcribe_audio(file_path: str) -> str:
    try:
        audio, sample_rate = sf.read(file_path)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
            sf.write(file_path, audio, sample_rate)
        return model.transcribe([file_path], batch_size=1)[0]
    except Exception as e:
        raise RuntimeError(f"Error during transcription: {e}")

def translate_text(text, target_language='en', source_language='hi', api_key='your-api-key'):
    url = "https://api.mymemory.translated.net/get"
    params = {
        'q': text,
        'langpair': f"{source_language}|{target_language}",
        'key': api_key
    }
    response = requests.get(url, params=params)
    return response.json().get('responseData', {}).get('translatedText', "")

def get_question_embedding(question: str) -> np.ndarray:

    embedding = sentence_model.encode(question)  # Generate embedding
    if not isinstance(embedding, np.ndarray):
        embedding = np.array(embedding)  # Ensure embedding is an np.ndarray
    return embedding

def load_text_chunks(pkl_path):
    """
    Load text chunks from a .pkl file.

    """
    try:
        with open(pkl_path, "rb") as f:
            text_chunks = pickle.load(f)
        return text_chunks
    except FileNotFoundError:
        raise RuntimeError(f"Pickle file {pkl_path} not found.")
    except Exception as e:
        raise RuntimeError(f"Error loading text chunks from {pkl_path}: {e}")

def diagnose_docstore(docstore):
    """
    Diagnose the structure and available methods of a docstore object.

    """
    print("Docstore Diagnostic Information:")
    print(f"Type of docstore: {type(docstore)}")
    print("\nAvailable attributes:")
    for attr in dir(docstore):
        print(attr)
    
    print("\nTrying to access dictionary:")
    try:
        print(f"Is this a dictionary? {isinstance(docstore, dict)}")
        print(f"Dictionary keys: {list(docstore.keys())[:10] if isinstance(docstore, dict) else 'N/A'}")
    except Exception as e:
        print(f"Error accessing dictionary: {e}")
    
    print("\nTrying to print the object directly:")
    try:
        print(docstore)
    except Exception as e:
        print(f"Error printing object: {e}")

def retrieve_chunks(index, question_embedding, text_chunks, top_k=2):
    """
    Diagnostic retrieve chunks function
    """
    if not isinstance(question_embedding, np.ndarray):
        question_embedding = np.array(question_embedding)
    if question_embedding.ndim == 1:
        question_embedding = np.expand_dims(question_embedding, axis=0)

    try:
        # Perform FAISS search
        distances, indices = index.search(question_embedding, top_k)

        # Diagnose the docstore
        print("\nDocstore Diagnosis:")
        diagnose_docstore(text_chunks[0])

        retrieved_chunks = [] 
        for idx in indices[0]:
            # Get the UUID for this index
            uuid = text_chunks[1].get(idx)
            print(f"\nProcessing index {idx}, UUID: {uuid}")
            
            if uuid:
                try:

                    print("Attempting to retrieve document content...")
                    
                    # Various methods to try accessing the document
                    if hasattr(text_chunks[0], '_dict'):
                        document = text_chunks[0]._dict.get(uuid)
                        print("Retrieved via _dict method")
                    elif isinstance(text_chunks[0], dict):
                        document = text_chunks[0].get(uuid)
                        print("Retrieved via dict method")
                    else:
                        document = None
                        print("Could not retrieve document")
                    
                    # Extract page content
                    if document:
                        page_content = document.get('page_content') if isinstance(document, dict) else getattr(document, 'page_content', "Content not found")
                        retrieved_chunks.append(page_content)
                        print(f"Successfully retrieved content: {page_content[:100]}...")
                
                except Exception as doc_error:
                    print(f"Error retrieving document for UUID {uuid}: {doc_error}")

        return retrieved_chunks

    except Exception as e:
        print(f"Error retrieving chunks: {e}")
        return []

def prepare_context(retrieved_chunks):
    """
    Prepare a string context for the LLM using retrieved text chunks.

    """
    return "\n".join(retrieved_chunks)


def query_cohere(context: str, question: str) -> str:
    url = "https://api.cohere.com/v1/generate"
    headers = {
        "Authorization": f"Bearer {COHERE_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    prompt = f"""You are an expert information retrieval assistant. 

Available Context:
{context}

Question: {question}

Instructions:
- Carefully analyze the provided context
- If the context contains relevant information, provide a precise and informative answer
- If the information is partial or insufficient, attempt to provide the most relevant information available
- If absolutely no relevant information exists, explain why you cannot answer

Detailed Answer:"""

    data = {
        "model": COHERE_MODEL,
        "prompt": prompt,
        "max_tokens": 300,
        "temperature": 0.7,
        "k": 0,
        "p": 0.75,
        "stop_sequences": ["\n\n"]
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        response_data = response.json()
        answer = response_data.get("generations", [{}])[0].get("text", "").strip()
        
        return answer
    
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Cohere API error: {e}")
    except ValueError as e:
        raise RuntimeError(f"Error parsing Cohere response: {e}")

@app.post("/rag-pipeline/")
async def rag_pipeline(file: UploadFile):
    try:
        if file.content_type not in ["audio/mpeg", "audio/wav", "audio/x-wav"]:
            raise HTTPException(status_code=400, detail="Invalid audio format. Please upload a valid audio file.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        # Step 1: Transcribe audio
        transcription = transcribe_audio(temp_file_path)

        # Step 2: Translate transcription
        translated_question = translate_text(transcription, api_key="266f0c55837af3a24e12")

        # Step 3: Retrieve relevant chunks
        question_embedding = get_question_embedding(translated_question)
        question_embedding = np.expand_dims(question_embedding, axis=0)  # Ensure it's a 2D array for FAISS
        
        text_chunks = load_text_chunks(pkl_path)
        retrieved_chunks = retrieve_chunks(index, question_embedding, text_chunks)
        # Step 4: Query Cohere LLM
        context = "\n".join(retrieved_chunks)
        answer = query_cohere(context, translated_question)
        print("Generated Answer:", answer)
       
        os.remove(temp_file_path)

        return {
            "transcription": transcription,
            "translated_question": translated_question,
            "retrieved_chunks": retrieved_chunks,
            "answer": answer,
            "text chunks": text_chunks
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

