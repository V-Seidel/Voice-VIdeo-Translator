from typing import Dict, List 
import os

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Application settings class for the API.
    """

    # Audio conversion settings
    data_path: str = 'data'
    video_link: str = 'https://www.youtube.com/watch?v=Wq5eO0VeC9U'
    freq: int = 16000
    filename: str = 'youtube_audio'

    # Txt file settings
    raw_text: str = 'raw_portuguese_text'
    corrected_text: str = 'corrected_english_text'

    # Model settings
    huggingface_speech_to_text_model: str = 'jonatasgrosman/wav2vec2-xls-r-1b-portuguese'
    huggingface_spellings_model: str = 'google/flan-t5-large'
    open_ai_api_key: str 

settings = Settings(_env_file=".env")
