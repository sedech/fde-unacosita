"""
Ingesta de documentos en ChromaDB con embeddings Gemini — PE3
"""

import ast
import os
import time
from pathlib import Path
from typing import Optional

import chromadb
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHROMA_BASE  = PROJECT_ROOT / "data" / "chroma"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL = "models/text-embedding-004"

CHUNK_CONFIGS = {
    "small": {"tam": 200, "overlap": 50},
    "large": {"tam": 500, "overlap": 100},
}


def _configurar_gemini():
    genai.configure(api_key=GEMINI_API_KEY)


def _chunkar(texto: str, tam: int, overlap: int) -> list[str]:
    chunks, inicio = [], 0
    while inicio < len(texto):
        fin = min(inicio + tam, len(texto))
        fragmento = texto[inicio:fin].strip()
        if fragmento:
            chunks.append(fragmento)
        inicio += tam - overlap
    return chunks


def _embed_batch(textos: list[str]) -> list[list[float]]:
    embeddings = []
    for texto in textos:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=texto,
            task_type="retrieval_document",
        )
        embeddings.append(result["embedding"])
    return embeddings


def _leer_documento(path: Path) -> tuple[dict[str, str], list[str]]:
    """Devuelve ({seccion: texto}, roles)."""
    secciones: dict[str, str] = {}
    roles: list[str] = ["cliente", "vendedor", "admin"]
    seccion_actual = "general"

    for linea in path.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if linea.startswith("— Sección:"):
            seccion_actual = linea.split("— Sección:", 1)[1].strip()
            secciones.setdefault(seccion_actual, "")
        elif linea.startswith("— Metadatos: roles ="):
            parte = linea.split("roles =", 1)[1].strip()
            try:
                roles = ast.literal_eval(parte)
            except (ValueError, SyntaxError):
                pass
        elif linea:
            secciones[seccion_actual] = secciones.get(seccion_actual, "") + linea + "\n"

    return secciones, roles


def ingestar(
    carpeta: str = "data/docs",
    config: Optional[str] = None,
) -> chromadb.Collection:
    """
    Lee los .txt de `carpeta`, genera embeddings Gemini y los guarda en ChromaDB.
    Usa config 'small' o 'large' (override de CHUNK_CONFIG env var).
    Retorna la colección ChromaDB.
    """
    _configurar_gemini()

    config = config or os.getenv("CHUNK_CONFIG", "small")
    if config not in CHUNK_CONFIGS:
        raise ValueError(f"CHUNK_CONFIG debe ser 'small' o 'large', recibido: {config!r}")

    tam     = CHUNK_CONFIGS[config]["tam"]
    overlap = CHUNK_CONFIGS[config]["overlap"]

    docs_dir = PROJECT_ROOT / carpeta
    chroma_path = CHROMA_BASE / config

    client = chromadb.PersistentClient(path=str(chroma_path))
    coleccion = client.get_or_create_collection(
        name="fde_unacosita",
        metadata={"hnsw:space": "cosine"},
    )

    total_fragmentos = 0

    for archivo in sorted(docs_dir.glob("*.txt")):
        t0 = time.monotonic()
        documento = archivo.stem
        secciones, roles = _leer_documento(archivo)

        ids, embeddings_list, textos, metadatas = [], [], [], []

        for seccion, texto in secciones.items():
            if not texto.strip():
                continue
            chunks = _chunkar(texto, tam, overlap)
            for i, chunk in enumerate(chunks):
                chunk_id = f"{documento}__{seccion}__{config}__{i}"
                ids.append(chunk_id)
                textos.append(chunk)
                metadatas.append({
                    "documento": documento,
                    "seccion":   seccion,
                    "config":    config,
                    "rol_cliente":  "cliente"  in roles,
                    "rol_vendedor": "vendedor" in roles,
                    "rol_admin":    "admin"    in roles,
                })

        if not ids:
            continue

        embeddings_list = _embed_batch(textos)

        coleccion.upsert(
            ids=ids,
            embeddings=embeddings_list,
            documents=textos,
            metadatas=metadatas,
        )

        ms = int((time.monotonic() - t0) * 1000)
        print(f"  {archivo.name}: {len(ids)} fragmentos ({config}) — {ms} ms")
        total_fragmentos += len(ids)

    print(f"  Total ({config}): {total_fragmentos} fragmentos en {chroma_path}")
    return coleccion
