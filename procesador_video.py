import os
import cv2
from PIL import Image
import torch
import numpy as np
import chromadb
from chromadb.config import Settings
from transformers import BlipProcessor, BlipForConditionalGeneration
import datetime
import hashlib

def generar_captions(frames, processor, model, device="cpu"):
    """
    Genera descripciones para cada frame utilizando el modelo BLIP
    
    Args:
        frames: Lista de rutas a los frames
        processor: BLIP processor inicializado
        model: BLIP model inicializado
        device: Dispositivo de cómputo (cuda o cpu)
        
    Returns:
        List: Lista de tuplas (frame_path, caption)
    """
    captions = []
    for frame_path in frames:
        image = Image.open(frame_path).convert("RGB")
        inputs = processor(image, return_tensors="pt").to(device)
        output = model.generate(**inputs)
        caption = processor.batch_decode(output, skip_special_tokens=True)[0]
        captions.append((frame_path, caption))
    return captions

def guardar_datos_video(video_path, frames_paths, captions, collection_name=None):
    """
    Guarda metadatos del video, frames y captions en ChromaDB
    
    Args:
        video_path: Ruta al video
        frames_paths: Lista de rutas a los frames
        captions: Lista de tuplas (frame_path, caption)
        collection_name: Nombre opcional para la colección
        
    Returns:
        str: Nombre de la colección creada
    """
    # Crear nombre de colección basado en el hash del video si no se proporciona
    if collection_name is None:
        with open(video_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        collection_name = f"video_{file_hash[:10]}"
    
    # Inicializar ChromaDB
    os.makedirs("./chroma_db", exist_ok=True)
    chroma_client = chromadb.Client(Settings(persist_directory="./chroma_db"))
    
    # Crear o recuperar la colección
    try:
        collection = chroma_client.get_collection(name=collection_name)
        print(f"Usando colección existente: {collection_name}")
    except:
        collection = chroma_client.create_collection(name=collection_name)
        print(f"Creando nueva colección: {collection_name}")
    
    # Preparar datos para guardar
    documents = []
    metadatas = []
    ids = []
    
    # Guardar información de cada frame
    for i, (frame_path, caption) in enumerate(captions):
        # Obtener timestamp del nombre del archivo
        tiempo = int(frame_path.split("_")[-1].split("s.jpg")[0])
        
        # Preparar documento con la descripción
        documents.append(caption)
        
        # Preparar metadata con información relevante
        metadatas.append({
            "tiempo": tiempo,
            "path": frame_path,
            "tipo": "frame"
        })
        
        # ID único para el documento
        ids.append(f"frame_{i}")
    
    # Guardar en ChromaDB
    if documents:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"✅ Guardados {len(documents)} frames con sus descripciones en ChromaDB")
    
    return collection_name

def procesar_video_completo(video_path, blip_processor, blip_model, device="cpu"):
    """
    Procesa un video extrayendo frames, generando captions y guardando en ChromaDB
    
    Args:
        video_path: Ruta al archivo de video
        blip_processor: Procesador BLIP inicializado
        blip_model: Modelo BLIP inicializado
        device: Dispositivo de cómputo (cuda o cpu)
        
    Returns:
        dict: Información sobre el procesamiento
    """
    print("🎬 Procesando video:", video_path)
    
    # 1. Extraer frames cada 5 segundos
    frames_paths = []
    print(f"✅ Frames extraídos: {len(frames_paths)}")
    
    # 2. Generar captions para cada frame
    if frames_paths:
        captions = generar_captions(frames_paths, blip_processor, blip_model, device)
        print(f"✅ Captions generados para {len(captions)} frames")
        
        # 3. Guardar en ChromaDB
        collection_name = guardar_datos_video(video_path, frames_paths, captions)
        
        return {
            "frames": frames_paths,
            "captions": captions,
            "collection_name": collection_name,
            "success": True
        }
    else:
        print("❌ No se pudieron extraer frames del video")
        return {
            "frames": [],
            "captions": [],
            "collection_name": None,
            "success": False
        }

def buscar_informacion_visual(pregunta, collection_name, top_k=3):
    """
    Busca frames relevantes a una pregunta usando ChromaDB
    
    Args:
        pregunta: Pregunta del usuario
        collection_name: Nombre de la colección en ChromaDB
        top_k: Número de resultados a devolver
        
    Returns:
        dict: Resultados de la búsqueda
    """
    try:
        # Inicializar ChromaDB
        chroma_client = chromadb.Client(Settings(persist_directory="./chroma_db"))
        
        # Obtener la colección
        collection = chroma_client.get_collection(name=collection_name)
        
        # Realizar búsqueda
        resultados = collection.query(
            query_texts=[pregunta],
            n_results=top_k
        )
        
        # Formatear resultados
        frames_info = []
        for i in range(len(resultados["ids"][0])):
            frame_id = resultados["ids"][0][i]
            metadata = resultados["metadatas"][0][i]
            texto = resultados["documents"][0][i]
            
            frames_info.append({
                "id": frame_id,
                "tiempo": metadata["tiempo"],
                "path": metadata["path"],
                "caption": texto
            })
        
        return {
            "success": True, 
            "frames": frames_info
        }
    except Exception as e:
        print(f"Error al buscar información visual: {str(e)}")
        return {
            "success": False,
            "frames": [],
            "error": str(e)
        }

def generar_contexto_visual(pregunta, info_video):
    """
    Genera contexto visual para responder a una pregunta sobre un video
    
    Args:
        pregunta: Pregunta del usuario
        info_video: Información del video procesado
        
    Returns:
        str: Texto con contexto visual para responder
    """
    collection_name = info_video.get("collection_name")
    if not collection_name:
        return ""
    
    # Buscar frames relevantes
    resultados = buscar_informacion_visual(pregunta, collection_name)
    if not resultados["success"] or not resultados["frames"]:
        return ""
    
    # Generar contexto con las descripciones de los frames
    contexto = "INFORMACIÓN VISUAL DEL VIDEO (procesada por un sistema de captioning de imágenes):\n\n"
    
    for i, frame in enumerate(resultados["frames"]):
        tiempo = frame["tiempo"]
        descripcion = frame["caption"]
        contexto += f"[Minuto {tiempo//60}:{tiempo%60:02d}] {descripcion}\n"
    
    # Agregar instrucciones explícitas para GPT
    contexto += "\nInstrucciones: Usa la información visual anterior para responder preguntas sobre lo que aparece en el video. Esta información ha sido generada automáticamente a partir de frames extraídos del video. No debes mencionar que no puedes ver videos, ya que la información visual ya ha sido procesada para ti."
    
    return contexto
