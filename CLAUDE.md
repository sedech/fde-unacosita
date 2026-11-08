# Contexto del proyecto — fde-unacosita

Proyecto del curso **Forward Deployed Engineer** (FDE).  
Cada entrega (PE1, PE2, PE3…) se arma en Claude Chat y se integra aquí.

## Stack
- Python 3.11+
- Groq SDK (`llama-3.3-70b-versatile`)
- Pydantic v2 (validación de entrada/salida)
- SQLAlchemy (DB, se expande en PE3)
- python-dotenv

## Comandos frecuentes

```bash
# Activar entorno
.venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Correr pruebas PE2
cd src && python ../scripts/ejecutar_5_consultas.py
```

## Convenciones
- Commits: `PE<n>: descripción breve` sin Co-Authored-By
- El autor de commits es siempre `sedech`
- Los archivos de cada entrega van en `src/` (módulos) y `scripts/` (runners)
- El prompt del motor está en `src/prompts/`

## Variables de entorno (.env)
- `GROQ_API_KEY` — clave de la API de Groq
- `DATABASE_URL` — SQLite por defecto (`sqlite:///./fde.db`)
- `GROQ_MODEL` — modelo a usar; `llama-3.3-70b-versatile` es el del curso pero requiere plan pago; en free tier usar `qwen/qwen3.8-27b`
- `NEGOCIO_NOMBRE` — nombre del negocio (default: `Ferretería UnaCosita`)
