from fastapi import FastAPI, UploadFile, HTTPException
import torch
import soundfile as sf
import nemo.collections.asr as nemo_asr
import tempfile
import os

# Load the NeMo ASR model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "ai4b_indicConformer_hi.nemo"

try:
    model = nemo_asr.models.EncDecCTCModel.restore_from(restore_path=MODEL_PATH)
    model.freeze()
    model = model.to(DEVICE)
    model.cur_decoder = "ctc"
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

# Initialize FastAPI app
app = FastAPI()

def transcribe_audio(file_path: str, language_id: str = "hi") -> str:
    """
    Transcribe audio using NeMo ASR model.

    """
    try:
        # Load the audio file
        audio, sample_rate = sf.read(file_path)

        # Check if the audio is stereo
        if len(audio.shape) > 1:
            # Convert to mono by averaging the channels
            audio = audio.mean(axis=1)

            # Save the mono audio back to a temporary file
            sf.write(file_path, audio, sample_rate)

        transcription = model.transcribe([file_path], batch_size=1)[0]
        return transcription
    except Exception as e:
        raise RuntimeError(f"Error during transcription: {e}")

@app.post("/transcribe/")
async def transcribe_endpoint(file: UploadFile):
    """
    API endpoint to accept an audio file and return its transcription.

    """
    try:
        # Validate file type
        if file.content_type not in ["audio/mpeg", "audio/wav", "audio/x-wav"]:
            raise HTTPException(status_code=400, detail="Invalid audio format. Please upload a valid audio file.")

        # Save the file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        # Transcribe the audio file
        transcription = transcribe_audio(temp_file_path)

        # Cleanup temporary file
        os.remove(temp_file_path)

        return {"transcription": transcription}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
