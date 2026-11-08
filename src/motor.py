"""
Motor LLM base — PE2
Recibe una consulta de precio/stock, la envía a Groq (llama-3.3-70b-versatile),
valida la respuesta con Pydantic y devuelve métricas reales de tokens, costo y latencia.
"""

import json
import os
import time
from pathlib import Path
from typing import Union

from dotenv import load_dotenv
from groq import Groq, APIConnectionError, RateLimitError, APIStatusError
from pydantic import ValidationError

from modelos import Canal, Entrada, ErrorValidacion, Salida

load_dotenv()

# ── Configuración ──────────────────────────────────────────────────────────────
GROQ_API_KEY    = os.getenv("GROQ_API_KEY")
MODELO          = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
NEGOCIO_NOMBRE  = os.getenv("NEGOCIO_NOMBRE", "Ferretería UnaCosita")

# Precios publicados Groq — llama-3.3-70b-versatile (USD por millón de tokens)
PRECIO_INPUT_POR_MILLON  = 0.59
PRECIO_OUTPUT_POR_MILLON = 0.79

PROMPTS_DIR = Path(__file__).parent / "prompts"

# Catálogo mínimo de ejemplo (en PE3 esto viene de la DB / RAG)
CATALOGO_EJEMPLO = """
- Tornillo cabeza hexagonal 1/4 x 1" (caja x50): $2.800
- Tornillo autoperforante 8 x 1" (caja x100): $1.950
- Llave inglesa 10" Stanley: $8.400
- Cinta métrica 5m Truper: $3.200
- Martillo carpintero 20oz: $6.100
- Pintura látex blanco 4L Sherwin: $12.500
- Lija al agua grano 220 (pliego): $450
- Manguera polietileno 1/2" x 25m: $9.800
"""


def _cargar_prompt(nombre: str) -> str:
    return (PROMPTS_DIR / nombre).read_text(encoding="utf-8")


def _calcular_costo(tokens_entrada: int, tokens_salida: int) -> float:
    return round(
        (tokens_entrada / 1_000_000) * PRECIO_INPUT_POR_MILLON
        + (tokens_salida / 1_000_000) * PRECIO_OUTPUT_POR_MILLON,
        8,
    )


def _llamar_groq(prompt_sistema: str, consulta_usuario: str) -> tuple[str, int, int, int]:
    """
    Llama a Groq y devuelve (texto_respuesta, tokens_entrada, tokens_salida, latencia_ms).
    Propaga excepciones de API para que el motor las capture.
    """
    cliente = Groq(api_key=GROQ_API_KEY)
    inicio = time.monotonic()

    respuesta = cliente.chat.completions.create(
        model=MODELO,
        messages=[
            {"role": "system", "content": prompt_sistema},
            {"role": "user",   "content": consulta_usuario},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    latencia_ms = int((time.monotonic() - inicio) * 1000)
    texto       = respuesta.choices[0].message.content
    tok_entrada = respuesta.usage.prompt_tokens
    tok_salida  = respuesta.usage.completion_tokens

    return texto, tok_entrada, tok_salida, latencia_ms


def _parsear_y_validar(
    texto_json: str,
    tokens_entrada: int,
    tokens_salida: int,
    latencia_ms: int,
) -> Salida:
    """Parsea el JSON y lo valida contra Salida. Lanza ValidationError si falla."""
    datos = json.loads(texto_json)
    datos["tokens_entrada"] = tokens_entrada
    datos["tokens_salida"]  = tokens_salida
    datos["costo_usd"]      = _calcular_costo(tokens_entrada, tokens_salida)
    datos["latencia_ms"]    = latencia_ms
    return Salida(**datos)


def procesar(entrada_raw: dict) -> Union[Salida, ErrorValidacion]:
    """
    Punto de entrada principal del motor.

    Parámetros
    ----------
    entrada_raw : dict con claves consulta, canal, negocio_id

    Retorna
    -------
    Salida validada, o ErrorValidacion si fallan los dos intentos.
    """
    # 1. Validar entrada
    entrada = Entrada(**entrada_raw)

    # 2. Construir prompt
    plantilla      = _cargar_prompt("consulta_precio.txt")
    prompt_sistema = plantilla.format(
        negocio_nombre    = NEGOCIO_NOMBRE,
        canal             = entrada.canal.value,
        contexto_catalogo = CATALOGO_EJEMPLO,
        consulta          = entrada.consulta,
    )

    # 3. Llamar al modelo — hasta 2 intentos
    for intento in range(1, 3):
        try:
            texto, tok_in, tok_out, lat_ms = _llamar_groq(prompt_sistema, entrada.consulta)
        except APIConnectionError as e:
            return ErrorValidacion(
                campo_fallido = "conexion",
                detalle       = f"Sin conexión a la API de Groq: {e}",
                consulta      = entrada.consulta,
                intento       = intento,
            )
        except RateLimitError as e:
            return ErrorValidacion(
                campo_fallido = "rate_limit",
                detalle       = f"Límite de uso alcanzado: {e}",
                consulta      = entrada.consulta,
                intento       = intento,
            )
        except APIStatusError as e:
            return ErrorValidacion(
                campo_fallido = "api_error",
                detalle       = f"Error de la API ({e.status_code}): {e.message}",
                consulta      = entrada.consulta,
                intento       = intento,
            )

        # 4. Validar respuesta
        try:
            salida = _parsear_y_validar(texto, tok_in, tok_out, lat_ms)
            return salida
        except (ValidationError, json.JSONDecodeError, KeyError) as e:
            campo = "json" if isinstance(e, json.JSONDecodeError) else str(e).split("\n")[1] if "\n" in str(e) else str(e)[:60]
            print(f"  [intento {intento}] Validación fallida — campo: {campo}")
            if intento == 2:
                return ErrorValidacion(
                    campo_fallido = campo,
                    detalle       = str(e),
                    consulta      = entrada.consulta,
                    intento       = intento,
                )
            # reintento automático

    # No debería llegar acá, pero por seguridad:
    return ErrorValidacion(
        campo_fallido = "desconocido",
        detalle       = "Falló después de 2 intentos sin excepción conocida",
        consulta      = entrada.consulta,
        intento       = 2,
    )
