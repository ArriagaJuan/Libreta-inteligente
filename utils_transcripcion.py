import os
import hashlib
import openai

# Asegúrate de tener tu API key en el entorno
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_transcripcion_path(file_path):
    """Obtiene una ruta única para la transcripción del archivo."""
    with open(file_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    return f"./transcripciones/{file_hash}.txt"

def obtener_o_transcribir_audio(file_path):
    """Transcribe el archivo si no existe ya su transcripción."""
    os.makedirs("transcripciones", exist_ok=True)
    path_txt = get_transcripcion_path(file_path)

    if os.path.exists(path_txt):
        with open(path_txt, "r", encoding="utf-8") as f:
            return f.read()

    with open(file_path, "rb") as f:
        transcript = openai.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )
    with open(path_txt, "w", encoding="utf-8") as f:
        f.write(transcript.text)
    return transcript.text

def extraer_audio_con_ffmpeg(video_path, output_audio_path="audio_extraido.wav"):
    import ffmpeg
    try:
        (
            ffmpeg
            .input(video_path)
            .output(output_audio_path, format='wav', acodec='pcm_s16le', ac=1, ar='16000')
            .overwrite_output()
            .run(quiet=True)
        )
        return output_audio_path
    except ffmpeg.Error:
        return None
