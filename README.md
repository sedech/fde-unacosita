# fde-unacosita

Proyecto del curso **Forward Deployed Engineer** — Ferretería UnaCosita.

Motor LLM base con validación Pydantic y medición de tokens/costo/latencia usando Groq.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
cp .env.example .env        # completar GROQ_API_KEY
```

## Correr pruebas PE2

```bash
cd src
python ../scripts/ejecutar_5_consultas.py
```

## Estructura

```
src/
  modelos.py          # Pydantic: Entrada, Salida, ErrorValidacion
  motor.py            # Motor LLM (Groq llama-3.3-70b-versatile)
  db.py               # Conexión SQLAlchemy
  prompts/
    consulta_precio.txt
scripts/
  ejecutar_5_consultas.py
```
