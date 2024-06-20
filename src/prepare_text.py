import os
import logging
from openai import OpenAI

from settings import settings

logging.basicConfig(level=logging.INFO)

template = """Contexto: Transcrição de um vídeo. Correção ortográfica: certifique-se de que todas as palavras estão escritas corretamente sem alterar o significado ou adicionar palavras novas. Corrija a ortografia do seguinte texto:

Exemplo 1:
Entrada: "Olá tudu bem vindos ao nosso canal onde falamos sobre tecnologea."
Saída: "Olá, tudo bem? Bem-vindos ao nosso canal onde falamos sobre tecnologia."

Exemplo 2:
Entrada: "Este video é para explicar como voce pode melhorar suas habilidades de programassão."
Saída: "Este vídeo é para explicar como você pode melhorar suas habilidades de programação."

Exemplo 3:
Entrada: "Hoje nos vamos ver como criar um site usando HTML, CSS e JavaScrip."
Saída: "Hoje, nós vamos ver como criar um site usando HTML, CSS e JavaScript."

Agora, corrija o texto abaixo, e após isso traduza para o inglês:"""

def load_txt_file(data_path: str,
                    filename: str) -> str:
    
    """
    Load text from a txt file.
    
    Args:
        data_path (str): Path to save the audio.
        filename (str): Name of the audio file.
    
    Returns:
        str: Text from the txt file.
    """
    
    try:
        with open(f'{data_path}/{filename}.txt', 'r') as f:
            text = f.read()
            logging.info('Text loaded successfully')
        return text
    except Exception as e:
        logging.error('Error loading text:', e)
        return -1


def save_txt_file(data_path: str,
                    filename: str,
                    text: str) -> int:
    
    """
    Save text to a txt file.
    
    Args:
        data_path (str): Path to save the audio.
        filename (str): Name of the audio file.
        text (str): Text to save.
    
    Returns:
        None
    """
    
    try:
        with open(f'{data_path}/{filename}.txt', 'w') as f:
            f.write(text)
            logging.info('Text saved successfully')
            return 0
    except Exception as e:
        logging.error('Error saving text:', e)
        return -1

def correct_text(text: str) -> str:
    """
    Correct the text.
    
    Args:
        text (str): Text to correct.
    
    Returns:
        str: Corrected text.
    """
    
    client = OpenAI(
        api_key=settings.open_ai_api_key,
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": template
            },
            {
                "role": "assistant",
                "content": text
            }
        ],
        model="gpt-3.5-turbo",
    )

    logging.info('Text corrected successfully')

    return(chat_completion.choices[0].message.content)

def prepare_text_pipeline(data_path: str,
                            input_raw_filename: str,
                            output_correct_filename:str) -> int:
    """
    Pipeline to correct text.
    
    Returns:
        None
    """
    text = load_txt_file(data_path, filename=input_raw_filename)
    corrected_text = correct_text(text)
    save_txt_file(data_path, filename=output_correct_filename, text=corrected_text)

    return 0