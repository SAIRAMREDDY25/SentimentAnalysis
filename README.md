# Sentiment Analysis

This repository contains a **Sentiment Analysis Bot** built with **Streamlit**, **AWS Transcribe**, **NLTK VADER** sentiment analysis, and **OpenAI**. The bot enables users to upload audio files, converts the audio to text using AWS Transcribe, and performs sentiment analysis on the transcribed text. It also allows users to interact with an OpenAI-based chatbot for detailed conversation insights and analysis.

## Features

- **Audio File Upload**: Upload `.wav` or `.mp3` files to analyze.
- **AWS Transcribe Integration**: Automatically converts audio to text via AWS Transcribe.
- **Sentiment Analysis**: Provides sentiment scores for the transcribed text using VADER from NLTK.
- **Keyword Highlighting**: Highlights specific keywords within the transcribed text.
- **Conversational Chatbot**: Uses OpenAI to analyze and discuss the conversation sentiment and context.
- **Streamlit UI**: User-friendly interface for uploading audio, viewing transcripts, and interacting with the chatbot.

## Setup and Installation

### Prerequisites

- Python 3.7 or later
- [Streamlit](https://streamlit.io/)
- [AWS Account](https://aws.amazon.com/) with S3 and Transcribe services enabled
- [OpenAI API Key](https://beta.openai.com/)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Environment Variables

Create a `.env` file in the project root and add the following environment variables:

```bash
openai_api_key=your_openai_api_key
aws_access_key_id=your_aws_access_key
aws_secret_access_key=your_aws_secret_key
```

### Configure AWS S3 Bucket

Set up an S3 bucket in AWS to store audio files and configure permissions to allow AWS Transcribe access.

### NLTK Data

The application downloads the VADER lexicon from NLTK for sentiment analysis:

```python
import nltk
nltk.download('vader_lexicon')
```

## Usage

1. **Run the Application**:
   ```bash
   streamlit run app.py
   ```
2. **Upload Audio File**:
   - Select an `.mp3` or `.wav` file to analyze.
   - The file is uploaded to S3 and processed by AWS Transcribe.
3. **View Transcript**:
   - Once the transcription completes, the text is displayed.
4. **Analyze Sentiment**:
   - The sentiment analysis results are shown as a JSON object.
5. **Highlight Keywords**:
   - Enter keywords to be highlighted in the transcript.
6. **Interact with Chatbot**:
   - Use the chatbot to get insights into the conversation sentiment, agent and customer behavior, and conversation context.

## File Structure

- `app.py`: Main application file for Streamlit
- `requirements.txt`: Python dependencies
- `.env`: Environment variables
- `tempDir/`: Temporary storage for uploaded audio files

## Security

**Important**: Do not expose your API keys or AWS credentials. Use a `.env` file and keep it out of version control by adding it to `.gitignore`.

## License

This project is licensed under the MIT License.
