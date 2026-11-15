"""
Script de demostración PE3 — RAG híbrido con filtro de permisos.
Correr desde la raíz del proyecto:
    python scripts/ejecutar_pe3.py
"""

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag.ingestar import ingestar
from rag.buscar import buscar
from rag.responder import responder

SEP = "─" * 90
ANCHO = 90


def titulo(texto: str):
    print(f"\n{'=' * ANCHO}")
    print(f"  {texto}")
    print(f"{'=' * ANCHO}\n")


def seccion(texto: str):
    print(f"\n{SEP}")
    print(f"  {texto}")
    print(SEP)


# ── 1. Ingesta en ambas configs ──────────────────────────────────────────────
titulo("PE3 · UnaCosita · RAG híbrido con permisos")

print("▶ Ingesta config=small")
col_small = ingestar(config="small")
print(f"\n▶ Ingesta config=large")
col_large = ingestar(config="large")

# ── 2. Evaluación top-k sobre las 10 preguntas ──────────────────────────────
seccion("Evaluación top-k · 10 preguntas de prueba")

preguntas_path = Path(__file__).resolve().parent.parent / "data" / "preguntas_prueba.json"
preguntas = json.loads(preguntas_path.read_text(encoding="utf-8"))

for config_eval in ("small", "large"):
    print(f"\n  Config: {config_eval}")
    print(f"  {'#':<4} {'Tipo':<13} {'Rol':<10} {'Esperado':<26} {'Hit?':<6} Pregunta")
    print(f"  {'-'*4} {'-'*13} {'-'*10} {'-'*26} {'-'*6} {'-'*30}")

    hits = 0
    for p in preguntas:
        esperado = p["documento_esperado"]
        resultados = buscar(p["pregunta"], rol=p["rol"], k=4, config=config_eval)
        docs_encontrados = [r["documento"] for r in resultados]

        if esperado is None:
            hit = len(docs_encontrados) == 0 or resultados[0]["puntaje"] < 0.3
            hit_str = "✅ vacío" if hit else "⚠️ trajo"
        else:
            hit = esperado in docs_encontrados
            hit_str = "✅" if hit else "❌"
            if hit:
                hits += 1

    normales = [p for p in preguntas if p["tipo"] == "normal"]
    pct = hits / len(normales) * 100 if normales else 0

    # Reimpresión con datos
    hits = 0
    for p in preguntas:
        esperado = p["documento_esperado"]
        resultados = buscar(p["pregunta"], rol=p["rol"], k=4, config=config_eval)
        docs_encontrados = [r["documento"] for r in resultados]

        if esperado is None:
            sin_ev = not resultados or resultados[0]["puntaje"] < 0.3
            hit_str = "✅" if sin_ev else "⚠️"
        else:
            hit = esperado in docs_encontrados
            hit_str = "✅" if hit else "❌"
            if hit:
                hits += 1

        print(f"  {p['id']:<4} {p['tipo']:<13} {p['rol']:<10} {str(esperado or '—'):<26} {hit_str:<6} {p['pregunta'][:40]}")

    pct = hits / len(normales) * 100 if normales else 0
    print(f"\n  → Top-k hit rate (preguntas normales, config={config_eval}): {hits}/{len(normales)} = {pct:.0f}%")

# ── 3. Híbrida vs vectorial: 2 casos donde BM25 ayuda ───────────────────────
seccion("BM25 mejora vectorial · 2 consultas exactas")

casos_exactos = [
    ("Tornillo 1/4 x 1", "cliente"),
    ("Distribuidora Norte contado", "vendedor"),
]
for consulta, rol in casos_exactos:
    resultados = buscar(consulta, rol=rol, k=4, config="small")
    print(f"\n  Consulta: {consulta!r}  (rol={rol})")
    for r in resultados[:3]:
        print(f"    [{r['documento']} / {r['seccion']}] puntaje={r['puntaje']:.4f}")
        print(f"    {r['texto'][:80]}")

# ── 4. Respuesta con fuentes ─────────────────────────────────────────────────
seccion("Respuesta con fuentes (requiere_humano=False)")

caso_con_fuente = {
    "consulta":    "¿Cuánto sale la caja de tornillos 1/4?",
    "canal":       "telegram",
    "negocio_id":  "ferreteria_demo",
    "rol":         "cliente",
}
print(f"\n  Consulta: {caso_con_fuente['consulta']}")
res = responder(caso_con_fuente)
if isinstance(res, dict) and not res.get("requiere_humano"):
    print(f"  Respuesta    : {res.get('respuesta', '')}")
    print(f"  Intención    : {res.get('intencion', '')}")
    print(f"  Confianza    : {res.get('confianza', '')}")
    print(f"  Fuentes      : {res.get('fuentes', [])}")
    print(f"  Tokens       : {res.get('tokens_entrada')} in / {res.get('tokens_salida')} out")
    print(f"  Latencia     : {res.get('latencia_ms')} ms")
else:
    print(f"  Resultado    : {res}")

# ── 5. Caso requiere_humano=True ─────────────────────────────────────────────
seccion("Caso sin evidencia (requiere_humano=True)")

caso_sin_ev = {
    "consulta":   "¿Tienen cemento Portland?",
    "canal":      "telegram",
    "negocio_id": "ferreteria_demo",
    "rol":        "cliente",
}
print(f"\n  Consulta: {caso_sin_ev['consulta']}")
res2 = responder(caso_sin_ev)
print(f"  requiere_humano : {res2.get('requiere_humano')}")
print(f"  motivo          : {res2.get('motivo')}")
print(f"  fuentes         : {res2.get('fuentes')}")

# ── 6. Tests de permisos ─────────────────────────────────────────────────────
seccion("Tests de permisos por rol (pytest)")

tests_path = str(Path(__file__).resolve().parent.parent / "tests")
resultado_pytest = subprocess.run(
    [sys.executable, "-m", "pytest", tests_path, "-v", "--tb=short", "--no-header"],
    capture_output=False,
    text=True,
)
print()
sys.exit(0 if resultado_pytest.returncode == 0 else resultado_pytest.returncode)
