import os
import sys
import json
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts import Prompt
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.extractors import QuestionsAnsweredExtractor, TitleExtractor, KeywordExtractor
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Definir bloques por categoría
CATEGORIAS = {
    "reconocimiento": ["Reconocimiento Oficial"],
    "formulacion1": ["Formulación de Proyectos"],
    "formulacion2": ["Formulación de Proyectos"],
    "formulacion3": ["Formulación de Proyectos"],
    "licitaciones": ["Licitaciones"]
}

# Obtener bloque desde argumento
if len(sys.argv) < 2:
    print("Error: Debes indicar el nombre del bloque (reconocimiento, formulacion1, etc.)")
    sys.exit(1)

bloque = sys.argv[1]
if bloque not in CATEGORIAS:
    print(f"Error: Bloque '{bloque}' no válido. Opciones: {list(CATEGORIAS.keys())}")
    sys.exit(1)

print(f"✅ Ejecutando indexador para el bloque: {bloque}\n")

# Leer metadata
with open("metadata.json", "r", encoding="utf-8") as f:
    metadata_dict = json.load(f)

# Filtrar documentos según la categoría
docs = []
for filename in os.listdir("docs"):
    if filename.endswith(".pdf"):
        nombre_base = os.path.splitext(filename)[0]
        meta = next((m for m in metadata_dict if m["nombre"] == nombre_base), None)
        if meta and meta.get("categoria") in CATEGORIAS[bloque]:
            docs.append(filename)

if not docs:
    print(f"No se encontraron documentos para el bloque '{bloque}'.")
    sys.exit(0)

print(f"Documentos encontrados: {docs}\n")

# Cargar prompt base
with open("prompt.txt", "r", encoding="utf-8") as f:
    prompt_text = f.read()
text_qa_template = Prompt(prompt_text)

# Cargar documentos
documents = SimpleDirectoryReader(input_files=[f"docs/{d}" for d in docs]).load_data()
for doc in documents:
    doc.metadata = next((m for m in metadata_dict if m["nombre"] == doc.metadata.get("file_name").replace(".pdf", "")), {})

# Preparar componentes del pipeline
client = OpenAI()
embed_model = OpenAIEmbedding(client=client)
parser = SentenceSplitter()
extractors = [
    TitleExtractor(nodes=1),
    KeywordExtractor(keywords=10, mode="flat"),
    QuestionsAnsweredExtractor(questions=3),
]
pipeline = IngestionPipeline(transformations=[parser, *extractors, embed_model])

# Ejecutar pipeline
nodes = pipeline.run(documents=documents)

# Guardar en carpeta de índice
output_dir = f"indices/{bloque}"
storage_context = StorageContext.from_defaults(persist_dir=output_dir)
index = VectorStoreIndex(nodes, storage_context=storage_context)
index.storage_context.persist()

print(f"\n📁 Índice del bloque '{bloque}' guardado en: {output_dir}")
