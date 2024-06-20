import os
import librosa
import soundfile as sf
import logging

from pytube import YouTube
from pydub import AudioSegment

logging.basicConfig(level=logging.INFO)

def download_audio(video_link: str,
                    data_path: str,
                    filename: str) -> int:
    
    """
    Download audio from a YouTube video link.

    Args:
        video_link (str): YouTube video link.
        data_path (str): Path to save the audio.
        filename (str): Name of the audio file.

    Returns:
        None
    """
    
    if filename.split('.')[-1] != 'mp4':
        filename += '.mp4'
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    try:
        yt = YouTube(video_link)
        yt.streams.filter(only_audio=True).first().download(output_path=data_path, filename=filename)
        logging.info('Audio downloaded successfully')
        return 0
    except Exception as e:
        logging.error('Error downloading audio:', e)
        return -1

def convert_audio_to_wav(data_path: str,
                        filename: str) -> int:
        
    """
    Convert audio from mp4 to wav format.

    Args:
        data_path (str): Path to save the audio.
        filename (str): Name of the audio file.
    
    Returns:
        None
    """

    try:
        audio = AudioSegment.from_file(f'{data_path}/{filename}.mp4')
        audio.export(f'{data_path}/{filename}.wav', format='wav')
        logging.info('Audio converted to wav successfully')
        return 0
    except Exception as e:
        logging.error('Error converting audio to wav:', e)
        return -1

def resample_audio(data_path: str,
                    freq: int,
                    filename: str) -> int:
    
    """
    Resample audio to a specific frequency.

    Args:
        data_path (str): Path to save the audio.
        freq (int): Frequency to resample the audio.
        filename (str): Name of the audio file.
    
    Returns:
        None
    """

    try:
        y, sr = librosa.load(f'{data_path}/{filename}.wav', sr=freq)
        sf.write(f'{data_path}/{filename}.wav', y, freq)
        logging.info('Audio resampled successfully')
        return 0
    except Exception as e:
        logging.error('Error resampling audio:', e)
        return -1
    

def prepare_video_pipeline(video_link: str,
                            data_path: str,
                            filename: str,
                            freq: int) -> int:
    
    """
    Prepare the video pipeline.

    Args:
        video_link (str): YouTube video link.
        data_path (str): Path to save the audio.
        filename (str): Name of the audio file.
        freq (int): Frequency to resample the audio.
    
    Returns:
        None
    """

    download_audio(video_link, data_path, filename)
    convert_audio_to_wav(data_path, filename)
    resample_audio(data_path, freq, filename)
    return 0