"""
Responder con contexto RAG — PE3
Usa buscar() para recuperar fragmentos y llama al motor de PE2 con prompt enriquecido.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from groq import APIConnectionError, RateLimitError, APIStatusError

import time
from groq import Groq
from modelos import Entrada, ErrorValidacion
from motor import _parsear_y_validar, _calcular_costo
from rag.buscar import buscar

load_dotenv()

PROJECT_ROOT   = Path(__file__).resolve().parent.parent.parent
PROMPTS_DIR    = PROJECT_ROOT / "src" / "prompts"
NEGOCIO_NOMBRE = os.getenv("NEGOCIO_NOMBRE", "Ferretería UnaCosita")

UMBRAL_CONFIANZA_RAG = 0.0  # filtro real de relevancia está en buscar.py (similitud coseno ≥ 0.70)


def _cargar_prompt(nombre: str) -> str:
    return (PROMPTS_DIR / nombre).read_text(encoding="utf-8")


def _construir_contexto(fragmentos: list[dict]) -> str:
    partes = []
    for f in fragmentos:
        partes.append(
            f"[{f['documento']} / {f['seccion']}]\n{f['texto']}"
        )
    return "\n\n".join(partes)


def responder(entrada_raw: dict) -> dict:
    """
    Recibe entrada con claves: consulta, canal, negocio_id, rol (default 'cliente').
    Devuelve dict con campos de Salida + fuentes + requiere_humano.
    En caso de error retorna ErrorValidacion.
    """
    rol = entrada_raw.get("rol", "cliente")
    entrada_sin_rol = {k: v for k, v in entrada_raw.items() if k != "rol"}

    # 1. Validar entrada con Pydantic (reutiliza contrato PE2)
    entrada = Entrada(**entrada_sin_rol)

    # 2. Recuperar fragmentos con filtro de permisos
    fragmentos = buscar(entrada.consulta, rol=rol, k=4)

    # 3. Sin evidencia suficiente → requiere_humano
    mejor_puntaje = fragmentos[0]["puntaje"] if fragmentos else 0.0
    if not fragmentos or mejor_puntaje < UMBRAL_CONFIANZA_RAG:
        return {
            "requiere_humano": True,
            "motivo":          "sin evidencia suficiente en los documentos",
            "fuentes":         [],
        }

    # 4. Prompt enriquecido con contexto RAG
    plantilla = _cargar_prompt("rag_respuesta.txt")
    contexto_rag = _construir_contexto(fragmentos)
    prompt_sistema = plantilla.format(
        negocio_nombre = NEGOCIO_NOMBRE,
        canal          = entrada.canal.value,
        contexto_rag   = contexto_rag,
        consulta       = entrada.consulta,
    )

    # 5. Llamar al LLM con prompt RAG — hasta 2 intentos
    groq_key = os.getenv("GROQ_API_KEY")
    modelo   = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    for intento in range(1, 3):
        try:
            cliente = Groq(api_key=groq_key)
            t0 = time.monotonic()
            resp = cliente.chat.completions.create(
                model=modelo,
                messages=[
                    {"role": "system", "content": prompt_sistema},
                    {"role": "user",   "content": entrada.consulta},
                ],
                temperature=0.1,
                max_tokens=256,
                response_format={"type": "json_object"},
            )
            lat_ms  = int((time.monotonic() - t0) * 1000)
            texto   = resp.choices[0].message.content
            tok_in  = resp.usage.prompt_tokens
            tok_out = resp.usage.completion_tokens
        except APIConnectionError as e:
            return ErrorValidacion(
                campo_fallido="conexion",
                detalle=f"Sin conexión a la API de Groq: {e}",
                consulta=entrada.consulta,
                intento=intento,
            )
        except RateLimitError as e:
            return ErrorValidacion(
                campo_fallido="rate_limit",
                detalle=f"Límite de uso alcanzado: {e}",
                consulta=entrada.consulta,
                intento=intento,
            )
        except APIStatusError as e:
            return ErrorValidacion(
                campo_fallido="api_error",
                detalle=f"Error de la API ({e.status_code}): {e.message}",
                consulta=entrada.consulta,
                intento=intento,
            )

        try:
            salida = _parsear_y_validar(texto, tok_in, tok_out, lat_ms)
            fuentes = list(dict.fromkeys(f["documento"] for f in fragmentos))
            return {
                **salida.model_dump(),
                "fuentes":         fuentes,
                "requiere_humano": False,
            }
        except Exception as e:
            if intento == 2:
                return ErrorValidacion(
                    campo_fallido="validacion_salida",
                    detalle=str(e),
                    consulta=entrada.consulta,
                    intento=intento,
                )

    return ErrorValidacion(
        campo_fallido="desconocido",
        detalle="Falló después de 2 intentos",
        consulta=entrada.consulta,
        intento=2,
    )
