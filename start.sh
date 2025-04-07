#!/bin/bash
set -e

echo "Iniciando servidor FastAPI..."
uvicorn main:app --host 0.0.0.0 --port $PORT
