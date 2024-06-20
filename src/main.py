import os
import logging
from types import GeneratorType

from settings import settings
from prepare_video import prepare_video_pipeline
from wave_to_string_module import wave_to_string_pipeline
from prepare_text import prepare_text_pipeline


logging.basicConfig(level=logging.INFO)


prepare_video_pipeline(video_link=settings.video_link,
                        data_path=settings.data_path,
                        filename=settings.filename,
                        freq=settings.freq)
wave_to_string_pipeline(data_path=settings.data_path,
                        filename_audio=settings.filename,
                        filename_raw_text=settings.raw_text,
                        freq=settings.freq)
prepare_text_pipeline(data_path=settings.data_path,
                        input_raw_filename=settings.raw_text,
                        output_correct_filename=settings.corrected_text)