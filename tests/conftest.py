import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
from rag.ingestar import ingestar
from rag.buscar import _get_collection


@pytest.fixture(scope="session", autouse=True)
def coleccion_lista():
    """Ingesta la colección 'small' si aún no existe o está vacía."""
    col = _get_collection("small")
    if col is None or col.count() == 0:
        ingestar(config="small")
