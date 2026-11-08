"""
Ejecuta las 5 consultas de prueba del PE2 e imprime la tabla de medición.
Correr desde la raíz del proyecto:
    python scripts/ejecutar_5_consultas.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from motor import procesar
from modelos import Salida, ErrorValidacion

CONSULTAS = [
    # (descripción, payload)
    (
        "Precio tornillo 1/4 (clara, en catálogo)",
        {"consulta": "Hola, ¿cuánto sale la caja de tornillos 1/4 x 1 pulgada?", "canal": "telegram", "negocio_id": "ferreteria_demo"},
    ),
    (
        "Stock llave inglesa (clara, en catálogo)",
        {"consulta": "¿Tienen llave inglesa 10 pulgadas Stanley?", "canal": "mostrador", "negocio_id": "ferreteria_demo"},
    ),
    (
        "Presupuesto pintura + lija (múltiples ítems)",
        {"consulta": "Necesito presupuesto de 2 latas de pintura blanca y 5 pliegos de lija 220", "canal": "telegram", "negocio_id": "ferreteria_demo"},
    ),
    (
        "Consulta confusa / ambigua",
        {"consulta": "ese fierrito que se usa para lo del techo, el que va en las esquinas", "canal": "telegram", "negocio_id": "ferreteria_demo"},
    ),
    (
        "Fuera de tema",
        {"consulta": "¿A qué hora abre el supermercado chino de la esquina?", "canal": "telegram", "negocio_id": "ferreteria_demo"},
    ),
]

SEPARADOR = "─" * 90

def main():
    print(f"\n{'=' * 90}")
    print("  Motor LLM Base — PE2 · UnaCosita · 5 consultas de prueba")
    print(f"{'=' * 90}\n")

    resultados = []

    for i, (descripcion, payload) in enumerate(CONSULTAS, start=1):
        print(f"[{i}] {descripcion}")
        print(f"    Consulta : {payload['consulta']}")
        print(f"    Canal    : {payload['canal']}")

        resultado = procesar(payload)

        if isinstance(resultado, Salida):
            print(f"    ✅ Validó")
            print(f"    Intención : {resultado.intencion.value}")
            print(f"    Producto  : {resultado.producto_detectado or '—'}")
            print(f"    Confianza : {resultado.confianza:.2f}")
            print(f"    Respuesta : {resultado.respuesta}")
            print(f"    Tokens    : {resultado.tokens_entrada} in / {resultado.tokens_salida} out")
            print(f"    Costo     : ${resultado.costo_usd:.6f} USD")
            print(f"    Latencia  : {resultado.latencia_ms} ms")
            resultados.append({
                "n": i, "valido": True,
                "tok_in": resultado.tokens_entrada,
                "tok_out": resultado.tokens_salida,
                "costo": resultado.costo_usd,
                "latencia": resultado.latencia_ms,
            })
        else:
            print(f"    ❌ Error — campo: {resultado.campo_fallido}")
            print(f"    Detalle  : {resultado.detalle}")
            resultados.append({
                "n": i, "valido": False,
                "tok_in": 0, "tok_out": 0, "costo": 0.0, "latencia": 0,
            })

        print(SEPARADOR)

    # Tabla resumen
    validos = [r for r in resultados if r["valido"]]
    print("\n  TABLA DE MEDICIÓN\n")
    print(f"  {'#':<4} {'¿Validó?':<10} {'Tok. entrada':<14} {'Tok. salida':<13} {'Costo USD':<14} {'Latencia ms'}")
    print(f"  {'-'*4} {'-'*10} {'-'*14} {'-'*13} {'-'*14} {'-'*12}")
    for r in resultados:
        valido_str = "Sí" if r["valido"] else "No"
        print(f"  {r['n']:<4} {valido_str:<10} {r['tok_in']:<14} {r['tok_out']:<13} {r['costo']:<14.6f} {r['latencia']}")

    if validos:
        prom_in  = sum(r["tok_in"]  for r in validos) / len(validos)
        prom_out = sum(r["tok_out"] for r in validos) / len(validos)
        prom_cos = sum(r["costo"]   for r in validos) / len(validos)
        prom_lat = sum(r["latencia"] for r in validos) / len(validos)
        print(f"  {'PROM':<4} {'—':<10} {prom_in:<14.0f} {prom_out:<13.0f} {prom_cos:<14.6f} {prom_lat:.0f}")

    print(f"\n  KPI técnico (meta PE1): extracción ≥ 90% de ítems correctos")
    print(f"  Latencia objetivo     : < 2000 ms por consulta")
    if validos:
        print(f"  Latencia promedio real: {prom_lat:.0f} ms  {'✅' if prom_lat < 2000 else '⚠️ supera la meta'}")
    print()

if __name__ == "__main__":
    main()
