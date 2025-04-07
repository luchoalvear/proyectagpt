import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts import Prompt
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.extractors import QuestionsAnsweredExtractor, TitleExtractor, KeywordExtractor
from dotenv import load_dotenv
import json

# Carga las variables de entorno (.env)
load_dotenv()

# 1. Cargar documentos PDF desde la carpeta /docs
documents = SimpleDirectoryReader("docs").load_data()

# 2. Leer prompt base desde prompt.txt
with open("prompt.txt", "r", encoding="utf-8") as f:
    prompt_text = f.read()
text_qa_template = Prompt(prompt_text)

# 3. Cargar metadata desde metadata.json
with open("metadata.json", "r", encoding="utf-8") as f:
    metadata_dict = json.load(f)

# 4. Asignar metadata a cada documento
for doc in documents:
    # Verificamos si el nombre del archivo está como string en doc.metadata
    filename = None

    if isinstance(doc.metadata, dict) and "file_name" in doc.metadata:
        filename = doc.metadata["file_name"]
    elif isinstance(doc.metadata, list):
        for item in doc.metadata:
            if isinstance(item, dict) and "file_name" in item:
                filename = item["file_name"]
                break

    doc.metadata = metadata_dict.get(filename, {})

# 5. Crear cliente moderno y modelo de embedding
client = OpenAI()
embed_model = OpenAIEmbedding(client=client)

# 6. Preparar el parser y los extractores
parser = SentenceSplitter()
extractors = [
    TitleExtractor(nodes=1),
    KeywordExtractor(keywords=10, mode="flat"),
    QuestionsAnsweredExtractor(questions=3),
]

# 7. Armar el pipeline
pipeline = IngestionPipeline(transformations=[parser, *extractors, embed_model])

# 8. Ejecutar el pipeline
nodes = pipeline.run(documents=documents)

# 9. Guardar en persistencia
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = VectorStoreIndex(nodes, storage_context=storage_context)
index.storage_context.persist()
