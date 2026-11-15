"""
Búsqueda híbrida vectorial + BM25 con filtro de permisos por rol — PE3
"""

import os
from pathlib import Path
from typing import Optional

import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHROMA_BASE  = PROJECT_ROOT / "data" / "chroma"
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL  = "models/text-embedding-004"
ROLES_VALIDOS    = {"cliente", "vendedor", "admin"}


def _configurar_gemini():
    genai.configure(api_key=GEMINI_API_KEY)


def _embed_query(consulta: str) -> list[float]:
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=consulta,
        task_type="retrieval_query",
    )
    return result["embedding"]


def _get_collection(config: str) -> Optional[chromadb.Collection]:
    chroma_path = CHROMA_BASE / config
    if not chroma_path.exists():
        return None
    try:
        client = chromadb.PersistentClient(path=str(chroma_path))
        return client.get_collection("fde_unacosita")
    except Exception:
        return None


def _rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


def buscar(
    consulta: str,
    rol: str,
    k: int = 4,
    config: str = "small",
) -> list[dict]:
    """
    Búsqueda híbrida: vectorial (Gemini) + BM25, fusión RRF, filtro de permisos.

    Retorna lista de dicts:
    {fragmento_id, documento, seccion, texto, puntaje, rol}
    """
    if rol not in ROLES_VALIDOS:
        raise ValueError(f"rol debe ser uno de {ROLES_VALIDOS}")

    coleccion = _get_collection(config)
    if coleccion is None:
        return []

    _configurar_gemini()

    filtro = {f"rol_{rol}": True}

    # ── Paso 1: todos los fragmentos accesibles para este rol (para BM25) ───
    todos = coleccion.get(where=filtro, include=["documents", "metadatas"])
    if not todos["ids"]:
        return []

    ids_todos      = todos["ids"]
    textos_todos   = todos["documents"]
    metas_todos    = todos["metadatas"]

    # ── Paso 2: búsqueda vectorial (top k*2) ────────────────────────────────
    query_emb  = _embed_query(consulta)
    n_vec      = min(k * 2, len(ids_todos))
    vec_res    = coleccion.query(
        query_embeddings=[query_emb],
        n_results=n_vec,
        where=filtro,
        include=["documents", "metadatas", "distances"],
    )
    vec_ids    = vec_res["ids"][0]          # top k*2 ids en orden de similitud
    vec_rank   = {id_: rank for rank, id_ in enumerate(vec_ids)}

    # ── Paso 3: BM25 sobre todos los accesibles ──────────────────────────────
    tokenizado = [t.lower().split() for t in textos_todos]
    bm25       = BM25Okapi(tokenizado)
    scores_bm  = bm25.get_scores(consulta.lower().split())

    bm25_ordenado = sorted(
        enumerate(scores_bm), key=lambda x: x[1], reverse=True
    )
    bm25_rank = {
        ids_todos[idx]: rank
        for rank, (idx, _) in enumerate(bm25_ordenado[:k * 2])
    }

    # ── Paso 4: fusión RRF ───────────────────────────────────────────────────
    candidatos = set(vec_rank) | set(bm25_rank)
    n_total    = len(ids_todos)
    rrf = {
        id_: _rrf_score(vec_rank.get(id_, n_total))
              + _rrf_score(bm25_rank.get(id_, n_total))
        for id_ in candidatos
    }

    top_ids = sorted(rrf, key=lambda x: rrf[x], reverse=True)[:k]

    # ── Paso 5: construir resultado ──────────────────────────────────────────
    idx_por_id = {id_: i for i, id_ in enumerate(ids_todos)}
    resultados = []
    for id_ in top_ids:
        idx  = idx_por_id[id_]
        meta = metas_todos[idx]
        resultados.append({
            "fragmento_id": id_,
            "documento":    meta.get("documento", ""),
            "seccion":      meta.get("seccion", ""),
            "texto":        textos_todos[idx],
            "puntaje":      round(rrf[id_], 6),
            "rol":          rol,
        })

    return resultados
