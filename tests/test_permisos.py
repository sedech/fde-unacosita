"""
Tests de control de acceso por rol — PE3
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag.buscar import buscar


def _documentos_encontrados(consulta: str, rol: str) -> list[str]:
    resultados = buscar(consulta, rol=rol, k=4)
    return [r["documento"] for r in resultados]


def test_cliente_no_ve_proveedores():
    """Un cliente no debe ver documentos de lista_proveedores."""
    docs = _documentos_encontrados("costo del proveedor tornillo margen ganancia", rol="cliente")
    assert "lista_proveedores" not in docs, (
        f"cliente NO debería ver lista_proveedores, encontró: {docs}"
    )


def test_vendedor_ve_proveedores():
    """Un vendedor sí puede ver lista_proveedores."""
    docs = _documentos_encontrados("costo Ferretodo tornillo proveedor", rol="vendedor")
    assert "lista_proveedores" in docs, (
        f"vendedor DEBERÍA ver lista_proveedores, encontró: {docs}"
    )


def test_admin_ve_condiciones():
    """Admin puede ver condiciones_comerciales."""
    docs = _documentos_encontrados("descuento volumen cuenta corriente", rol="admin")
    assert "condiciones_comerciales" in docs, (
        f"admin DEBERÍA ver condiciones_comerciales, encontró: {docs}"
    )


def test_vendedor_no_ve_condiciones():
    """Vendedor no puede ver condiciones_comerciales (solo admin)."""
    docs = _documentos_encontrados("descuento compras cuenta corriente empresas habilitadas", rol="vendedor")
    assert "condiciones_comerciales" not in docs, (
        f"vendedor NO debería ver condiciones_comerciales, encontró: {docs}"
    )


def test_cliente_ve_catalogo():
    """Cliente puede ver el catálogo público."""
    docs = _documentos_encontrados("precio tornillo 1/4 caja", rol="cliente")
    assert "catalogo_publico" in docs, (
        f"cliente DEBERÍA ver catalogo_publico, encontró: {docs}"
    )
