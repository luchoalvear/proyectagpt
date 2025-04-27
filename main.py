from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.query_engine import RetrieverQueryEngine

import os

app = FastAPI()

# Definir las carpetas de índices
INDICES_PATH = "./indices"

# Definir los bloques que vamos a cargar
BLOQUES = ["reconocimiento", "formulacion1", "formulacion2", "formulacion3", "licitaciones"]

# Diccionario para almacenar los query engines
query_engines = {}

# Cargar todos los índices al iniciar
for bloque in BLOQUES:
    try:
        persist_dir = os.path.join(INDICES_PATH, bloque)
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
        retriever = index.as_retriever()
        query_engine = RetrieverQueryEngine.from_args(retriever=retriever)
        query_engines[bloque] = query_engine
        print(f"✅ Índice '{bloque}' cargado correctamente.")
    except Exception as e:
        print(f"⚠️ Error cargando el índice '{bloque}': {e}")

# Modelo para la entrada
class Consulta(BaseModel):
    bloque: str
    pregunta: str

@app.get("/")
def read_root():
    return {"mensaje": "Servidor ProyectaGPT en funcionamiento."}

@app.post("/preguntar")
def preguntar(consulta: Consulta):
    bloque = consulta.bloque
    pregunta = consulta.pregunta

    if bloque not in query_engines:
        raise HTTPException(status_code=400, detail=f"Bloque '{bloque}' no encontrado.")
    
    query_engine = query_engines[bloque]
    respuesta = query_engine.query(pregunta)

    return {"respuesta": str(respuesta)}
