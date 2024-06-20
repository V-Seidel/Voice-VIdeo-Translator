import os
import logging
from types import GeneratorType

import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from settings import settings
from prepare_video import resample_audio

logging.basicConfig(level=logging.INFO)

tokenizer = Wav2Vec2Processor.from_pretrained(settings.huggingface_speech_to_text_model)
model = Wav2Vec2ForCTC.from_pretrained(settings.huggingface_speech_to_text_model)

def create_chunk_stream(data_path: str,
                        filename: str,
                        freq: int) -> GeneratorType:
    
    """
    Create a stream of audio chunks from a wav file.
    
    Args:
        data_path (str): Path to save the audio.
        filename (str): Name of the audio file.
        freq (int): Sampling frequency.
    
    Returns:
        None
    """

    sample_rate = librosa.get_samplerate(f'{data_path}/{filename}.wav')

    if sample_rate != freq:
        logging.info('Resampling audio because the frequency is different from the model')
        resample_audio(data_path, freq, filename)
    
    try:
        stream = librosa.stream(f'{data_path}/{filename}.wav', 
                                block_length=5, 
                                frame_length=freq, 
                                hop_length=freq)
        logging.info('Audio stream created successfully')
        return stream
    except Exception as e:
        logging.error('Error creating audio stream:', e)
        return -1
    

def audio_to_text(stream: GeneratorType,
                freq: int) -> str:
    
    """
    Convert audio chunks to text.
    
    Args:
        stream (GeneratorType): Audio stream.
    
    Returns
        str: Transcribed text.
    """

    text = ''
    for chunk in stream:
        input_values = tokenizer(chunk, return_tensors='pt', sampling_rate=freq).input_values
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = tokenizer.batch_decode(predicted_ids)[0]
        text += transcription
    return text


def save_text(text: str,
            filename: str) -> int:
    
    """
    Save transcribed text to a file.
    
    Args:
        text (str): Transcribed text.
        filename (str): Name of the file.
    
    Returns:
        None
    """

    try:
        with open(f'{settings.data_path}/{filename}.txt', 'w') as f:
            f.write(text)
        logging.info('Text saved successfully')
        return 0
    except Exception as e:
        logging.error('Error saving text:', e)
        return -1

def wave_to_string_pipeline(data_path: str,
                            filename_audio: str,
                            filename_raw_text: str,
                            freq: int) -> int:
    """
    Pipeline to convert audio to text.
    
    Returns:
        None
    """
    stream = create_chunk_stream(data_path, filename_audio, freq)
    text = audio_to_text(stream, freq)
    save_text(text, filename_raw_text)

    return 0

