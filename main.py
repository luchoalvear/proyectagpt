from fastapi import FastAPI
from pydantic import BaseModel
from llama_index.core import StorageContext, load_index_from_storage, ResponseMode
import os

app = FastAPI(
    title="ProyectaGPT - API",
    description="Servidor oficial de ProyectaGPT para consultas normativas",
    version="1.0.0"
)

# Diccionario global para guardar los índices cargados
indices = {}

# Ruta base donde están los índices
BASE_INDICES_PATH = "./indice"

# Lista de bloques disponibles
BLOQUES = ["formulacion1", "formulacion2", "formulacion3", "licitaciones", "reconocimiento"]

# Cargar todos los índices al iniciar el servidor
for bloque in BLOQUES:
    bloque_path = os.path.join(BASE_INDICES_PATH, bloque)
    if os.path.exists(bloque_path):
        storage_context = StorageContext.from_defaults(persist_dir=bloque_path)
        index = load_index_from_storage(storage_context)
        indices[bloque] = index
        print(f"✅ Índice '{bloque}' cargado correctamente.")
    else:
        print(f"⚠️ Advertencia: No se encontró la carpeta '{bloque_path}'")

# Modelo de datos para recibir consultas
class Consulta(BaseModel):
    pregunta: str
    bloque: str

# Endpoint para chequear que el servidor funcione
@app.get("/")
def read_root():
    return {"message": "Servidor Proyecta funcionando correctamente 🚀"}

# Endpoint para realizar preguntas
@app.post("/preguntar/")
async def preguntar(consulta: Consulta):
    if consulta.bloque not in indices:
        return {"error": f"Bloque '{consulta.bloque}' no encontrado. Bloques disponibles: {list(indices.keys())}"}
    
    index = indices[consulta.bloque]

    # Crear un query_engine personalizado para respuestas largas
    query_engine = index.as_query_engine(
        response_mode=ResponseMode.NO_TEXT_SUMMARIZATION,  # 🚀 No resumir, traer toda la info relevante
        similarity_top_k=5,  # 🔵 Traer los 5 fragmentos más relevantes (puedes ajustar este número)
        verbose=True  # Opcional: para ver logs en consola
    )

    respuesta = query_engine.query(consulta.pregunta)
    
    return {"respuesta": str(respuesta)}
