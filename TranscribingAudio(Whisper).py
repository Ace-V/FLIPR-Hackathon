#Code Successful

import streamlit as st
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch
import torchaudio
from pydub import AudioSegment
import tempfile
import os
import transformers
import sentence_transformers
from dotenv import load_dotenv
load_dotenv()


HF_TOKEN = os.getenv("HF_TOKEN")

@st.cache_resource
def load_model():
    model_id = "openai/whisper-tiny"
    processor = AutoProcessor.from_pretrained(model_id, token=HF_TOKEN)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, token=HF_TOKEN)
    model.eval()
    return processor, model

processor, model = load_model()

def convert_to_wav(uploaded_file):
    ext = uploaded_file.name.split('.')[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as temp_in:
        temp_in.write(uploaded_file.read())
        temp_in.flush()
        audio = AudioSegment.from_file(temp_in.name)
        wav_path = temp_in.name + ".wav"
        audio.export(wav_path, format="wav")
        return wav_path

import torchaudio
torchaudio.set_audio_backend("soundfile")

def transcribe_audio(file_path):
    import torchaudio
    import torch
    torchaudio.set_audio_backend("soundfile")

    waveform, sr = torchaudio.load(file_path)

    
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)


    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)

    
    inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        generated_ids = model.generate(
            inputs.input_features,
            forced_decoder_ids=processor.get_decoder_prompt_ids(language="en", task="transcribe")
        )

    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription




st.title("🎙️ Whisper Speech-to-Text (HF Transformers - Offline)")

uploaded_file = st.file_uploader("Upload audio or video file", type=["mp3", "wav", "m4a", "mp4", "mov", "mkv"])

if uploaded_file:
    with st.spinner("Processing..."):
        wav_path = convert_to_wav(uploaded_file)
        transcript = transcribe_audio(wav_path)
        os.remove(wav_path)

    st.success("Transcription complete!")
    st.text_area("Transcript:", transcript, height=300)
    st.download_button("Download .txt", transcript, file_name="transcript.txt")
