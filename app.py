import streamlit as st
import boto3
import time
import json
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
from pydub import AudioSegment
import os
from streamlit_chat import message
from openai import OpenAI
from dotenv import load_dotenv
import requests

# Download NLTK data
nltk.download('vader_lexicon')

load_dotenv()
api_key = os.getenv("openai_api_key")
client = OpenAI(api_key=api_key)

# AWS credentials
aws_access_key_id = 'aws_access_key_id'
aws_secret_access_key = 'aws_secret_access_key'

# Initialize AWS clients
transcribe_client = boto3.client('transcribe', region_name='ap-south-1',
                                 aws_access_key_id=aws_access_key_id,
                                 aws_secret_access_key=aws_secret_access_key)

# Function to convert mp3 to wav if needed
def convert_audio(file, file_path):
    if file_path.endswith('.mp3'):
        audio_segment = AudioSegment.from_mp3(file)
        audio_segment.export(file_path.replace('.mp3', '.wav'), format="wav")
        return file_path.replace('.mp3', '.wav')
    return file_path

# Function to upload audio file to S3 for AWS Transcribe
def upload_to_s3(file_path, bucket_name, object_name):
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id,
                      aws_secret_access_key=aws_secret_access_key)
    s3.upload_file(file_path, bucket_name, object_name)

# Function to start transcription job with AWS Transcribe
def start_transcription_job(s3_uri, job_name):
    response = transcribe_client.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={'MediaFileUri': s3_uri},
        MediaFormat='wav',
        LanguageCode='en-US',
    )
    return response

# Function to poll for the transcription result
def get_transcription_result(job_name):
    while True:
        status = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
        if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
            if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
                transcript_uri = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
                transcript = download_transcript(transcript_uri)
                return transcript
            else:
                raise ValueError("Transcription failed")
        time.sleep(10)  # Increased polling interval

# Download transcript
def download_transcript(transcript_uri):
    response = requests.get(transcript_uri)
    transcript_json = response.json()
    transcript_text = transcript_json['results']['transcripts'][0]['transcript']
    return transcript_text

# Function to analyze sentiment using VADER
def analyze_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)
    return sentiment_scores

# Highlight keywords in the transcript
def highlight_keywords(transcript, keywords):
    highlighted_transcript = transcript
    for keyword in keywords:
        highlighted_transcript = re.sub(f"({keyword})", r"*\1*", highlighted_transcript, flags=re.IGNORECASE)
    return highlighted_transcript

# OpenAI GPT chatbot with template and transcript-based conversation
def get_chatbot_response(prompt, context, transcript):
    conversation_template = f"""
    You are a sentiment analysis chatbot trained to analyze conversations between agents and customers.
    Below is the conversation transcript. Use it to answer questions about the sentiment, behavior, and overall tone of the interaction.

    Transcript: {transcript}

    You should provide insights into the following:
    1. The overall sentiment of the conversation.
    2. The customer’s sentiment (e.g., angry, happy, frustrated, neutral).
    3. The agent’s sentiment (e.g., empathetic, professional, rushed, neutral).
    4. The goal of the conversation (e.g., resolving an issue, providing information).
    5. A summary of the conversation.

    Please answer the following question based on this context:
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": conversation_template},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# Streamlit UI
st.title("Chat Message Sentiment Analysis Bot")

# Step 1: File uploader for audio
uploaded_file = st.file_uploader("Upload an audio file (wav or mp3)", type=["wav", "mp3"])

if uploaded_file is not None:
    # Step 2: Create tempDir if it doesn't exist
    os.makedirs("tempDir", exist_ok=True)

    # Step 3: Save the uploaded file locally
    file_name = uploaded_file.name
    file_path = os.path.join("tempDir", file_name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Step 4: Convert audio if needed
    audio_path = convert_audio(uploaded_file, file_path)

    # Step 5: Upload audio to S3
    st.write("Uploading audio to S3...")
    bucket_name = "sentim-1"
    object_name = f"transcriptions/{file_name}"
    upload_to_s3(audio_path, bucket_name, object_name)

    # Step 6: Start transcription job
    s3_uri = f"s3://{bucket_name}/{object_name}"
    job_name = f"transcription-job-{int(time.time())}"
    st.write(f"Starting transcription job: {job_name}")
    start_transcription_job(s3_uri, job_name)

    # Step 7: Poll for result and fetch transcript
    st.write("Waiting for transcription to complete...")
    try:
        transcript_text = get_transcription_result(job_name)
        st.write("Transcription completed! Here's the transcript:")
        st.text_area("Transcript", transcript_text, height=200)

        # Step 8: Overall Sentiment Analysis
        overall_sentiment = analyze_sentiment(transcript_text)
        st.write("Overall Sentiment:")
        st.json(overall_sentiment)

        # Step 9: Highlight specific keywords in the transcript
        keywords = st.text_input("Enter keywords to highlight (comma separated)", value="competition, pricing, service")
        if keywords:
            keywords_list = [keyword.strip() for keyword in keywords.split(",")]
            highlighted_transcript = highlight_keywords(transcript_text, keywords_list)
            st.write("Highlighted Transcript:")
            st.markdown(f"<pre>{highlighted_transcript}</pre>", unsafe_allow_html=True)

        # Step 10: Chatbot functionality with conversation-like responses
        if "messages" not in st.session_state:
            st.session_state.messages = []

        
        st.markdown(
    """
    <style>
    .chat-container {
        max-height: 400px; /* Set max height for the chat area */
        overflow-y: auto;  /* Enable scrolling for chat messages */
        padding-right: 10px;
        padding-bottom: 70px; /* Prevent chat messages from overlapping with input */
    }
    .input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        padding: 10px;
        background-color: #f9f9f9;
        z-index: 999;
        box-shadow: 0 -2px 5px rgba(0,0,0,0.1); /* Optional for better appearance */
    }
    </style>
    """, unsafe_allow_html=True
)


        # Display chat messages in a conversation-like format inside a scrollable container
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for i, chat in enumerate(st.session_state.messages):
            if chat["role"] == "user":
                message(chat["content"], is_user=True, key=f"user_{i}")
            else:
                message(chat["content"], key=f"bot_{i}")
        st.markdown('</div>', unsafe_allow_html=True)
        context = f"""
        You are a chatbot that assists with analyzing the sentiment and context of the conversation below.
        Respond in a friendly, conversational way. The conversation you're analyzing is:
       
        {transcript_text}
        """

        # Step 11: Input box for user message, fixed at bottom and cleared after submission
        def submit_input():
            user_input = st.session_state.input  # Get the input value
            # user_input = st.chat_input("Type your message here...")
            if user_input:
                st.session_state.messages.append({"role": "user", "content": user_input})  # Append user message
                response = get_chatbot_response(user_input, context, transcript_text)  # Get chatbot response
                st.session_state.messages.append({"role": "assistant", "content": response})  # Append bot message
                st.session_state.input = ""  # Clear input after submission

        # Fixed input box at the bottom for user input
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        st.text_input("You:", key="input", on_change=submit_input, placeholder="Type your message here...")
        st.markdown('</div>', unsafe_allow_html=True)

    except ValueError as e:
        st.error(str(e))
