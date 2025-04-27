import os
import json
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage

# Rutas
DOCS_PATH = "./docs/TXT"
INDICES_PATH = "./indices"
METADATA_PATH = "./metadata.json"

# Bloques definidos
BLOQUES = ["reconocimiento", "formulacion1", "formulacion2", "formulacion3", "licitaciones"]

# Cargar metadata global
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata_global = json.load(f)

def procesar_bloque(bloque, metadata_global):
    print(f"\n🔵 Procesando bloque: {bloque}...")

    documentos = []

    # Filtrar archivos que pertenecen a este bloque
    for nombre_archivo, atributos in metadata_global.items():
        if atributos.get("categoria", "").lower() == bloque.lower() or atributos.get("bloque", "").lower() == bloque.lower():
            ruta_archivo = os.path.join(DOCS_PATH, nombre_archivo.replace(".pdf", ".txt"))
            if os.path.exists(ruta_archivo):
                documentos.append(ruta_archivo)
            else:
                print(f"⚠️ No se encontró el archivo: {nombre_archivo.replace('.pdf', '.txt')}")

    if not documentos:
        print(f"⚠️ No se generaron documentos para el bloque '{bloque}'.")
        return

    # Leer documentos
    reader = SimpleDirectoryReader(input_files=documentos)
    chunks = reader.load_data()

    print(f"✅ {len(chunks)} documentos/chunks preparados para el bloque '{bloque}'.")

    # Crear índice
    index = VectorStoreIndex.from_documents(chunks)

    # Guardar índice
    persist_dir = os.path.join(INDICES_PATH, bloque)
    index.storage_context.persist(persist_dir=persist_dir)
    print(f"🔖 Índice '{bloque}' guardado en {persist_dir}")

# Procesar todos los bloques
def main():
    for bloque in BLOQUES:
        procesar_bloque(bloque, metadata_global)

    print("\n✅ Proceso de indexación finalizado.")

if __name__ == "__main__":
    main()
