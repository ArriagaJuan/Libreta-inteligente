import os
from PIL import Image
import torch
import chromadb
from chromadb.config import Settings

def generar_guardar_embeddings(frames_paths, bridge_processor, bridge_model, device, collection_name="video_frames"):
    """
    Genera embeddings para los frames extraídos y los guarda en ChromaDB
    
    Args:
        frames_paths: Lista con las rutas a los frames extraídos
        bridge_processor: Procesador BridgeTower inicializado
        bridge_model: Modelo BridgeTower inicializado
        device: Dispositivo de cómputo (cuda o cpu)
        collection_name: Nombre de la colección en ChromaDB
        
    Returns:
        collection: Objeto colección de ChromaDB
    """
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
    
    embeddings = []
    metadatas = []
    ids = []
    
    print(f"Procesando {len(frames_paths)} frames...")
    
    # Para cada frame, generar embedding
    for i, frame_path in enumerate(frames_paths):
        # Cargar imagen
        image = Image.open(frame_path)
        
        # Procesar imagen con BridgeTower
        inputs = bridge_processor(images=image, text="", return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = bridge_model(**inputs)
        
        # Obtener embedding de imagen
        image_embedding = outputs.image_embeds.mean(dim=1).cpu().numpy()
        
        # Extraer metadata (tiempo del frame)
        tiempo = int(frame_path.split("_")[-1].split("s.jpg")[0])
        
        # Guardar resultados
        embeddings.append(image_embedding[0].tolist())
        metadatas.append({"tiempo": tiempo, "path": frame_path})
        ids.append(f"frame_{i}")
    
    # Guardar en ChromaDB
    if embeddings:
        collection.add(
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print(f"✅ Guardados {len(embeddings)} embeddings en ChromaDB")
    else:
        print("❌ No se generaron embeddings")
    
    return collection


def buscar_similares(query_image_path, bridge_processor, bridge_model, device, collection_name="video_frames", n_results=5):
    """
    Busca frames similares a una imagen de consulta
    
    Args:
        query_image_path: Ruta a la imagen de consulta
        bridge_processor: Procesador BridgeTower inicializado
        bridge_model: Modelo BridgeTower inicializado
        device: Dispositivo de cómputo (cuda o cpu)
        collection_name: Nombre de la colección en ChromaDB
        n_results: Número de resultados a devolver
        
    Returns:
        resultados: Resultados de la búsqueda
    """
    # Inicializar ChromaDB
    chroma_client = chromadb.Client(Settings(persist_directory="./chroma_db"))
    
    # Obtener la colección
    try:
        collection = chroma_client.get_collection(name=collection_name)
    except:
        print(f"❌ No se encontró la colección: {collection_name}")
        return None
    
    # Cargar imagen de consulta
    image = Image.open(query_image_path)
    
    # Procesar imagen con BridgeTower
    inputs = bridge_processor(images=image, text="", return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = bridge_model(**inputs)
    
    # Obtener embedding de imagen
    query_embedding = outputs.image_embeds.mean(dim=1).cpu().numpy()[0].tolist()
    
    # Buscar en ChromaDB
    resultados = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    return resultados
