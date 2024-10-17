import streamlit as st
from moviepy.editor import VideoFileClip, CompositeAudioClip, AudioFileClip
from google.cloud import speech, texttospeech
import io
import os
import requests
from pydub import AudioSegment
import tempfile

from dotenv import load_dotenv
load_dotenv()

# Set up Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"

def extract_audio(video_file):
    """Extract audio from the uploaded video file."""
    video = VideoFileClip(video_file)
    audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    video.audio.write_audiofile(audio_file.name)
    video.close()
    return audio_file.name

def convert_to_mono(input_audio):
    """Convert the audio to mono channel."""
    audio = AudioSegment.from_wav(input_audio)
    mono_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    mono_audio = audio.set_channels(1)
    mono_audio.export(mono_audio_file.name, format="wav")
    return mono_audio_file.name

def transcribe_audio_with_timestamps(audio_file):
    """Transcribe audio with timestamps using Google Speech-to-Text."""
    client = speech.SpeechClient()
    resultSpeech = ""
    # Load audio file
    with io.open(audio_file, "rb") as f:
        content = f.read()

    # Set up the RecognitionAudio and RecognitionConfig
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code="en-US",
        enable_word_time_offsets=True
    )

    # Perform the speech recognition
    response = client.recognize(config=config, audio=audio)
    
    # Extract transcripts and timestamps
    last_end = 0
    for result in response.results:
        alternative = result.alternatives[0]
        for word_info in alternative.words:
            word = word_info.word
            start_time = word_info.start_time.total_seconds()
            end_time = word_info.end_time.total_seconds()
            resultSpeech += f"(Word: {word}, breakPoint: {start_time - last_end}), "
            last_end = end_time
    return resultSpeech

def correct_grammar(transcribed_text):
    """Correct grammar using Azure OpenAI."""
    azure_openai_key = os.getenv("AZURE_OPENAI_KEY")  # Load from .env
    azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")  # Load from .env
    headers = {
        "Content-Type": "application/json",
        "api-key": azure_openai_key
    }

    data = {
        "messages": [{"role": "user", "content": f"""
                    you will get a input of list of pairs in the format "Word: (word), breakPoint: (time)".
                    Using (word) from each line create a grammatically correct paragraph with punctuations then using this new paragraph replace the old word from given list with grammatically correct words and corresponding breakpoint.
                    Please add punctuations like comma or full stop or exclamation.
                    Input: '{transcribed_text}'.
                    Only return the final list of pair without any quotation or anything and nothing else in one line without newline.
                    """}],
        "max_tokens": len(transcribed_text)
    }
    response = requests.post(azure_openai_endpoint, headers=headers, json=data)

    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    else:
        return f"Error: {response.status_code} - {response.text}"

def convert_to_pairs(text):
    """Convert text to pairs of words and breakpoints."""
    pairs = []
    # Split the input string into individual entries
    entries = text.split('), (')
    
    # Process each entry to extract word and breakPoint
    for entry in entries:
        entry = entry.strip('() ')
        components = entry.split(',')
        word = components[0].split(':')[1].strip()
        break_point = float(components[1].split(':')[1].strip().rstrip(')'))
        pairs.append((word, break_point))
    return pairs

def convert_to_ssml_with_breaks(pairs):
    """Convert word-breakpoint pairs to SSML format."""
    ssml_output = '<speak><prosody rate="slow">'
    for word, break_point in pairs:
        if break_point > 0:
            ssml_output += f'<break time="{break_point}s"/>'
        ssml_output += f'{word} '
    ssml_output += '</prosody></speak>'
    return ssml_output

def synthesize_speech(input_ssml):
    """Synthesize speech using Google Text-to-Speech."""
    client = texttospeech.TextToSpeechClient()
    input_text = texttospeech.SynthesisInput(ssml=input_ssml)
    
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        ssml_gender=texttospeech.SsmlVoiceGender.MALE
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = client.synthesize_speech(
        input=input_text, voice=voice, audio_config=audio_config
    )

    audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    with open(audio_file.name, "wb") as out:
        out.write(response.audio_content)
    return audio_file.name

def combine_audio_with_video(video_file, new_audio_file, output_file):
    """Combine the new synthesized audio with the original video."""
    video = VideoFileClip(video_file)
    new_audio = AudioFileClip(new_audio_file)
    
    final_video = video.set_audio(new_audio)
    final_video.write_videofile(output_file, codec="libx264", audio_codec="aac")
    return output_file

# Streamlit UI
st.title("Video Audio Replacement with TTS")

uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "mov"])

if uploaded_video:
    video_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    with open(video_file_path, "wb") as f:
        f.write(uploaded_video.getbuffer())
    
    st.video(uploaded_video)

    # Step 1: Extract audio from video
    st.write("Extracting audio from video...")
    extracted_audio_path = extract_audio(video_file_path)
    st.write("Audio extracted successfully.")

    # Step 2: Convert audio to mono
    st.write("Converting audio to mono...")
    mono_audio_path = convert_to_mono(extracted_audio_path)
    st.write("Converted audio to mono.")

    # Step 3: Transcribe audio with timestamps
    st.write("Transcribing audio...")
    transcript = transcribe_audio_with_timestamps(mono_audio_path)
    st.write("Transcription completed.")

    # Step 4: Correct grammar
    st.write("Correcting grammar...")
    corrected_text = correct_grammar(transcript)
    st.write("Grammar corrected.")

    # Step 5: Convert corrected text to pairs and SSML
    st.write("Converting to SSML format...")
    word_pairs = convert_to_pairs(corrected_text)
    ssml_input = convert_to_ssml_with_breaks(word_pairs)

    # Step 6: Synthesize speech
    st.write("Synthesizing speech...")
    synthesized_audio_path = synthesize_speech(ssml_input)
    st.audio(synthesized_audio_path)
    
    # Step 7: Combine new audio with video
    st.write("Combining new audio with video...")
    final_video_path = combine_audio_with_video(video_file_path, synthesized_audio_path, "final_output.mp4")
    
    # Display the final video
    st.video(final_video_path)

    st.write("Process completed. You can download the video below:")
    st.download_button("Download Final Video", final_video_path, file_name="final_output.mp4")
