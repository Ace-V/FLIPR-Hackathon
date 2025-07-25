import streamlit as st
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import torchaudio
import tempfile
import os
import re
from datetime import datetime, timedelta
from dotenv import load_dotenv
import json

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

# Page config
st.set_page_config(page_title="AI Meeting Transcriber", page_icon="🎙️", layout="wide")

@st.cache_resource
def load_whisper_model():
    """Load Whisper model for transcription"""
    model_id = "openai/whisper-medium"
    processor = AutoProcessor.from_pretrained(model_id, token=HF_TOKEN)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, token=HF_TOKEN)
    model.eval()
    return processor, model

@st.cache_resource
def load_text_generation_model():
    """Load text generation model for meeting analysis"""
    model_name = "microsoft/DialoGPT-medium"
    # Alternative models you can try:
    # "facebook/blenderbot-400M-distill" - Good for conversational analysis
    # "microsoft/DialoGPT-large" - Larger model for better analysis
    # "google/flan-t5-large" - Instruction-following model
    
    try:
        text_generator = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=model_name,
            token=HF_TOKEN,
            max_length=512,
            pad_token_id=50256
        )
        return text_generator
    except Exception as e:
        st.error(f"Error loading text generation model: {e}")
        return None

@st.cache_resource
def load_summarization_model():
    """Load summarization model"""
    model_name = "facebook/bart-large-cnn"
    summarizer = pipeline("summarization", model=model_name, tokenizer=model_name, token=HF_TOKEN)
    return summarizer

@st.cache_resource 
def load_question_answering_model():
    """Load QA model for extracting specific information"""
    qa_model = pipeline(
        "question-answering",
        model="deepset/roberta-base-squad2",
        token=HF_TOKEN
    )
    return qa_model

@st.cache_resource
def load_sentiment_analyzer():
    """Load sentiment analysis model"""
    sentiment_analyzer = pipeline("sentiment-analysis", 
                                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                                token=HF_TOKEN)
    return sentiment_analyzer

# Load models
whisper_processor, whisper_model = load_whisper_model()
summarizer = load_summarization_model()
text_generator = load_text_generation_model()
qa_model = load_question_answering_model()
sentiment_analyzer = load_sentiment_analyzer()

import torchaudio

def convert_to_wav(uploaded_file):
    ext = uploaded_file.name.split('.')[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as temp_in:
        temp_in.write(uploaded_file.read())
        temp_in.flush()
        # If WAV, return as is
        if ext.lower() == "wav":
            return temp_in.name
        # Try to load with torchaudio and save as WAV
        waveform, sample_rate = torchaudio.load(temp_in.name)
        wav_path = temp_in.name + ".wav"
        torchaudio.save(wav_path, waveform, sample_rate)
        return wav_path

def transcribe_audio(file_path, chunk_length_sec=60):
    import torchaudio
    import math
    torchaudio.set_audio_backend("soundfile")

    waveform, sr = torchaudio.load(file_path)
    
    # Force mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if needed
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)
        sr = 16000

    # Chunking
    total_samples = waveform.shape[1]
    chunk_samples = int(chunk_length_sec * sr)
    num_chunks = math.ceil(total_samples / chunk_samples)

    transcripts = []
    for i in range(num_chunks):
        start = i * chunk_samples
        end = min((i + 1) * chunk_samples, total_samples)
        chunk_waveform = waveform[:, start:end]

        # Sometimes last chunk may be too short for Whisper model, skip if tiny
        if chunk_waveform.shape[1] < sr * 3:  # less than 3 seconds
            continue

        inputs = whisper_processor(chunk_waveform.squeeze(), sampling_rate=sr, return_tensors="pt")

        with torch.no_grad():
            generated_ids = whisper_model.generate(
                inputs.input_features,
                forced_decoder_ids=whisper_processor.get_decoder_prompt_ids(language="en", task="transcribe")
            )
        transcription = whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        transcripts.append(transcription)

    return "\n".join(transcripts)


def detect_speakers(transcript):
    """Simple speaker detection based on conversation patterns"""
    # Basic speaker detection using conversation cues
    lines = transcript.split('.')
    speakers = []
    current_speaker = "Speaker 1"
    speaker_count = 1
    
    conversation_markers = [
        "said", "mentioned", "asked", "replied", "responded", "stated",
        "according to", "as discussed by", "john", "mary", "sarah", "mike", "david"
    ]
    
    for line in lines:
        line = line.strip().lower()
        if any(marker in line for marker in conversation_markers):
            speaker_count += 1
            current_speaker = f"Speaker {min(speaker_count, 4)}"  # Max 4 speakers
        
        if line:
            speakers.append((current_speaker, line.capitalize()))
    
    return speakers

def extract_deadlines(text):
    """Extract deadlines and time-sensitive information"""
    deadline_patterns = [
        r'by (\w+day)',  # by Friday, Monday, etc.
        r'before (\w+day)',  # before Friday
        r'until (\w+day)',  # until Friday
        r'by (.+?) we need',  # by next week we need
        r'before (.+?) meeting',  # before next meeting
        r'deadline (.+?)(?:\.|,|$)',  # deadline next week
        r'due (.+?)(?:\.|,|$)',  # due tomorrow
        r'(\w+day) morning',  # Friday morning
        r'end of (.+?)(?:\.|,|$)',  # end of week
        r'next (.+?)(?:\.|,|$)',  # next week
    ]
    
    deadlines = []
    for pattern in deadline_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            deadline_text = match.group(1) if match.group(1) else match.group(0)
            context_start = max(0, match.start() - 50)
            context_end = min(len(text), match.end() + 50)
            context = text[context_start:context_end]
            deadlines.append({
                'deadline': deadline_text,
                'context': context.strip(),
                'urgency': 'high' if any(word in deadline_text.lower() for word in ['today', 'tomorrow', 'urgent']) else 'medium'
            })
    
    return deadlines

def extract_ownership(text):
    """Extract task ownership and responsibilities"""
    ownership_patterns = [
        r'(\w+) will (.+?)(?:\.|,|$)',  # John will handle this
        r'(\w+) should (.+?)(?:\.|,|$)',  # Mary should check
        r'let\'s have (\w+) (.+?)(?:\.|,|$)',  # let's have John lead
        r'(\w+) is responsible for (.+?)(?:\.|,|$)',  # John is responsible for
        r'(\w+) needs to (.+?)(?:\.|,|$)',  # Sarah needs to
        r'assign (\w+) to (.+?)(?:\.|,|$)',  # assign Mike to
        r'(\w+) can (.+?)(?:\.|,|$)',  # David can handle
    ]
    
    ownerships = []
    for pattern in ownership_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            person = match.group(1)
            task = match.group(2) if len(match.groups()) > 1 else match.group(1)
            context_start = max(0, match.start() - 30)
            context_end = min(len(text), match.end() + 30)
            context = text[context_start:context_end]
            ownerships.append({
                'person': person.title(),
                'task': task.strip(),
                'context': context.strip()
            })
    
    return ownerships

def analyze_tone_and_importance(text, speakers_data):
    """Analyze tone and importance of different segments"""
    analysis_results = []
    
    for speaker, content in speakers_data:
        if len(content) > 10:  # Only analyze substantial content
            # Sentiment analysis
            sentiment_result = sentiment_analyzer(content[:512])  # Limit input length
            sentiment = sentiment_result[0]['label']
            confidence = sentiment_result[0]['score']
            
            # Importance keywords
            importance_keywords = [
                'critical', 'important', 'urgent', 'priority', 'must', 'essential',
                'deadline', 'asap', 'immediately', 'crucial', 'key', 'vital'
            ]
            
            importance_score = sum(1 for keyword in importance_keywords if keyword in content.lower())
            importance_level = 'high' if importance_score >= 2 else 'medium' if importance_score >= 1 else 'low'
            
            analysis_results.append({
                'speaker': speaker,
                'content': content,
                'sentiment': sentiment,
                'sentiment_confidence': confidence,
                'importance': importance_level,
                'importance_score': importance_score
            })
    
    return analysis_results

def extract_information_with_qa(text, questions):
    """Extract specific information using QA model"""
    if not qa_model:
        return {}
    
    results = {}
    for question_key, question in questions.items():
        try:
            answer = qa_model(question=question, context=text[:2000])  # Limit context length
            if answer['score'] > 0.3:  # Only include confident answers
                results[question_key] = {
                    'answer': answer['answer'],
                    'confidence': answer['score']
                }
        except Exception as e:
            results[question_key] = {'answer': 'Unable to extract', 'confidence': 0.0}
    
    return results

def generate_meeting_analysis_with_hf(transcript, deadlines, ownership, speaker_analysis):
    """Generate comprehensive meeting analysis using Hugging Face models"""
    if not text_generator:
        return "Text generation model not available."
    
    # Create structured analysis using multiple approaches
    analysis_sections = {}
    
    # 1. Generate executive summary
    try:
        executive_prompt = f"Meeting Summary: {transcript[:800]}... \n\nKey points from this meeting:"
        exec_summary = text_generator(executive_prompt, max_length=200, num_return_sequences=1, temperature=0.7)
        analysis_sections['executive_summary'] = exec_summary[0]['generated_text'].replace(executive_prompt, "").strip()
    except:
        analysis_sections['executive_summary'] = "Unable to generate executive summary."
    
    # 2. Extract key information using QA model
    qa_questions = {
        'main_topic': 'What was the main topic of this meeting?',
        'key_decisions': 'What decisions were made in this meeting?',
        'next_steps': 'What are the next steps mentioned in this meeting?',
        'concerns_raised': 'What concerns or issues were raised in this meeting?'
    }
    
    qa_results = extract_information_with_qa(transcript, qa_questions)
    
    # 3. Create structured output
    structured_analysis = f"""
## 📋 Meeting Analysis Report

### 🎯 Executive Summary
{analysis_sections.get('executive_summary', 'Summary not available')}

### 🔍 Key Information Extracted
**Main Topic:** {qa_results.get('main_topic', {}).get('answer', 'Not clearly identified')}

**Key Decisions:** {qa_results.get('key_decisions', {}).get('answer', 'No clear decisions identified')}

**Next Steps:** {qa_results.get('next_steps', {}).get('answer', 'No specific next steps mentioned')}

**Concerns Raised:** {qa_results.get('concerns_raised', {}).get('answer', 'No major concerns identified')}

### ✅ Action Items & Ownership
"""
    
    if ownership:
        for task in ownership:
            structured_analysis += f"- **{task['person']}**: {task['task']}\n"
    else:
        structured_analysis += "- No clear task assignments identified\n"
    
    structured_analysis += "\n### 🕐 Deadlines & Time-Sensitive Items\n"
    
    if deadlines:
        for deadline in deadlines:
            urgency_icon = "🔴" if deadline['urgency'] == 'high' else "🟡"
            structured_analysis += f"{urgency_icon} **{deadline['deadline']}** - {deadline['context'][:100]}...\n"
    else:
        structured_analysis += "- No specific deadlines mentioned\n"
    
    structured_analysis += "\n### 📊 Meeting Insights\n"
    
    if speaker_analysis:
        positive_speakers = [s for s in speaker_analysis if s.get('sentiment') == 'POSITIVE']
        high_importance = [s for s in speaker_analysis if s.get('importance') == 'high']
        
        structured_analysis += f"- **Speakers with positive tone:** {len(positive_speakers)}\n"
        structured_analysis += f"- **High-importance segments:** {len(high_importance)}\n"
        
        if high_importance:
            structured_analysis += "\n**Most Important Points:**\n"
            for segment in high_importance[:3]:  # Top 3 important segments
                structured_analysis += f"- {segment['speaker']}: {segment['content'][:100]}...\n"
    
    return structured_analysis

def generate_summary(text, max_length=150):
    """Generate summary using BART model"""
    try:
        # Split text into chunks if too long
        max_chunk_length = 1024
        if len(text) > max_chunk_length:
            chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
            summaries = []
            for chunk in chunks:
                if len(chunk) > 50:  # Only summarize substantial chunks
                    summary = summarizer(chunk, max_length=max_length//len(chunks), min_length=30, do_sample=False)
                    summaries.append(summary[0]['summary_text'])
            return ' '.join(summaries)
        else:
            summary = summarizer(text, max_length=max_length, min_length=30, do_sample=False)
            return summary[0]['summary_text']
    except Exception as e:
        return f"Error in summarization: {str(e)}"

# Streamlit UI
st.title("🎙️ AI-Powered Meeting Transcriber & Analyzer")
st.markdown("Upload your meeting audio and get intelligent transcription, summarization, and task extraction!")

# Sidebar for settings
st.sidebar.header("Settings")
include_speaker_detection = st.sidebar.checkbox("Speaker Detection", value=True)
include_sentiment_analysis = st.sidebar.checkbox("Sentiment Analysis", value=True)
include_llm_analysis = st.sidebar.checkbox("Advanced LLM Analysis", value=True)
summary_length = st.sidebar.slider("Summary Length", 50, 300, 150)

# Main interface
uploaded_file = st.file_uploader(
    "Upload audio or video file", 
    type=["mp3", "wav", "m4a", "mp4", "mov", "mkv", "flac", "ogg"]
)

if uploaded_file:
    with st.spinner("🎯 Processing your meeting audio..."):
        # Step 1: Convert and transcribe
        wav_path = convert_to_wav(uploaded_file)
        transcript = transcribe_audio(wav_path)
        os.remove(wav_path)
    
    # Display results in tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📝 Transcript", "📊 Summary", "✅ Tasks & Deadlines", 
        "🎭 Speaker Analysis", "🤖 AI Insights"
    ])
    
    with tab1:
        st.subheader("Meeting Transcript")
        st.text_area("Raw Transcript:", transcript, height=300)
        st.download_button("📥 Download Transcript", transcript, file_name="meeting_transcript.txt")
    
    with tab2:
        st.subheader("Meeting Summary")
        with st.spinner("Generating summary..."):
            summary = generate_summary(transcript, max_length=summary_length)
        st.write(summary)
        st.download_button("📥 Download Summary", summary, file_name="meeting_summary.txt")
    
    with tab3:
        st.subheader("Tasks & Deadlines")
        
        # Extract deadlines and ownership
        deadlines = extract_deadlines(transcript)
        ownership = extract_ownership(transcript)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🕐 Deadlines Found:**")
            if deadlines:
                for deadline in deadlines:
                    urgency_color = "🔴" if deadline['urgency'] == 'high' else "🟡"
                    st.write(f"{urgency_color} **{deadline['deadline']}**")
                    st.write(f"*Context: {deadline['context']}*")
                    st.write("---")
            else:
                st.write("No specific deadlines detected.")
        
        with col2:
            st.write("**👥 Task Ownership:**")
            if ownership:
                for task in ownership:
                    st.write(f"**{task['person']}**: {task['task']}")
                    st.write(f"*Context: {task['context']}*")
                    st.write("---")
            else:
                st.write("No clear task assignments detected.")
    
    with tab4:
        if include_speaker_detection:
            st.subheader("Speaker Analysis")
            speakers_data = detect_speakers(transcript)
            
            if include_sentiment_analysis:
                with st.spinner("Analyzing tone and sentiment..."):
                    tone_analysis = analyze_tone_and_importance(transcript, speakers_data)
                
                for analysis in tone_analysis:
                    with st.expander(f"{analysis['speaker']} - {analysis['sentiment']} ({analysis['importance']} importance)"):
                        st.write(f"**Content:** {analysis['content']}")
                        st.write(f"**Sentiment:** {analysis['sentiment']} (confidence: {analysis['sentiment_confidence']:.2f})")
                        st.write(f"**Importance Level:** {analysis['importance']}")
            else:
                for speaker, content in speakers_data:
                    with st.expander(f"{speaker}"):
                        st.write(content)
        else:
            st.write("Speaker detection disabled.")
    
    with tab5:
        if include_llm_analysis:
            st.subheader("Advanced AI Analysis")
            
            with st.spinner("🤖 Generating comprehensive meeting analysis..."):
                # Prepare data for analysis
                speakers_data = detect_speakers(transcript) if include_speaker_detection else []
                deadlines = extract_deadlines(transcript)
                ownership = extract_ownership(transcript)
                
                speaker_analysis_data = []
                if include_sentiment_analysis:
                    speaker_analysis_data = analyze_tone_and_importance(transcript, speakers_data)
                
                # Generate comprehensive analysis using HF models
                hf_analysis = generate_meeting_analysis_with_hf(
                    transcript, deadlines, ownership, speaker_analysis_data
                )
                
                st.markdown(hf_analysis)
                st.download_button("📥 Download AI Analysis", hf_analysis, file_name="meeting_ai_analysis.txt")
        
        else:
            st.write("Advanced LLM analysis disabled.")
    
    # Summary metrics
    st.sidebar.header("Meeting Metrics")
    word_count = len(transcript.split())
    st.sidebar.metric("Word Count", word_count)
    st.sidebar.metric("Estimated Duration", f"{word_count // 150} minutes")
    
    if deadlines:
        st.sidebar.metric("Deadlines Found", len(deadlines))
    if ownership:
        st.sidebar.metric("Task Assignments", len(ownership))

else:
    # Welcome message
    st.markdown("""
    ### Welcome to the AI Meeting Transcriber! 🚀
    
    This advanced tool will help you:
    - 📝 **Transcribe** your meeting audio with high accuracy
    - 📊 **Summarize** key discussion points automatically  
    - ✅ **Extract** action items, deadlines, and task assignments
    - 🎭 **Analyze** speaker sentiment and importance levels
    - 🤖 **Generate** comprehensive meeting insights with AI
    
    **Supported formats:** MP3, WAV, M4A, MP4, MOV, MKV, FLAC, OGG
    
    **Get started:** Upload your meeting audio file above!
    
    **Note:** This app uses only open-source Hugging Face models - no API keys required for basic functionality!
    """)
    
    # Sample features showcase
    with st.expander("🌟 Key Features"):
        st.markdown("""
        **Smart Context Awareness:**
        - Automatic speaker identification and segmentation
        - Tone and sentiment analysis for each speaker
        - Importance level detection for different segments
        
        **Deadline Intelligence:**
        - Extracts time-sensitive information ("by Friday", "before next call")
        - Identifies urgency levels and context
        - Creates actionable deadline summaries
        
        **Ownership Detection:**
        - Identifies task assignments ("Let's have John lead this")
        - Maps responsibilities to team members
        - Tracks action item ownership
        
        **AI-Powered Analysis:**
        - Comprehensive meeting summaries using BART and DialoGPT
        - Question-answering for key information extraction
        - Decision tracking and next steps identification
        - All powered by open-source Hugging Face models
        """)
        
    with st.expander("🔧 Models Used"):
        st.markdown("""
        **Speech-to-Text:** OpenAI Whisper (tiny)
        **Summarization:** Facebook BART-large-CNN  
        **Text Generation:** Microsoft DialoGPT-medium
        **Question Answering:** Deepset RoBERTa-base-SQuAD2
        **Sentiment Analysis:** Cardiff NLP Twitter RoBERTa
        """)

# Add model loading status
if st.sidebar.button("🔄 Check Model Status"):
    st.sidebar.write("**Model Loading Status:**")
    st.sidebar.write(f"✅ Whisper: Loaded")
    st.sidebar.write(f"✅ Summarizer: Loaded") 
    st.sidebar.write(f"{'✅' if text_generator else '❌'} Text Generator: {'Loaded' if text_generator else 'Failed'}")
    st.sidebar.write(f"{'✅' if qa_model else '❌'} QA Model: {'Loaded' if qa_model else 'Failed'}")
    st.sidebar.write(f"✅ Sentiment: Loaded")