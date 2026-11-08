"""
Contrato de datos — PE2
Historia de usuario: consulta de precio y disponibilidad (canal Telegram / mostrador / web)
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class Canal(str, Enum):
    telegram  = "telegram"
    mostrador = "mostrador"
    web       = "web"


class Intencion(str, Enum):
    consulta_precio  = "consulta_precio"
    consulta_stock   = "consulta_stock"
    presupuesto      = "presupuesto"
    fuera_de_tema    = "fuera_de_tema"
    ambiguo          = "ambiguo"


class Entrada(BaseModel):
    consulta:    str   = Field(..., min_length=3, max_length=500,
                               description="Texto de la consulta del cliente o vendedor")
    canal:       Canal = Field(..., description="Canal de origen de la consulta")
    negocio_id:  str   = Field(..., min_length=1,
                               description="Identificador del negocio (tenant)")

    # Regla 1: la consulta no puede ser solo espacios o caracteres especiales
    @field_validator("consulta")
    @classmethod
    def consulta_no_vacia(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("La consulta no puede estar vacía o contener solo espacios")
        return v.strip()

    # Regla 2: negocio_id sin espacios (slug limpio)
    @field_validator("negocio_id")
    @classmethod
    def negocio_sin_espacios(cls, v: str) -> str:
        if " " in v:
            raise ValueError("negocio_id no puede contener espacios")
        return v.lower()


class Salida(BaseModel):
    respuesta:         str       = Field(..., min_length=1,
                                         description="Respuesta generada para el cliente")
    intencion:         Intencion = Field(..., description="Intención detectada en la consulta")
    producto_detectado: Optional[str] = Field(None,
                                              description="Producto mencionado en la consulta, si aplica")
    confianza:         float     = Field(..., ge=0.0, le=1.0,
                                         description="Confianza del modelo (0 a 1)")
    tokens_entrada:    int       = Field(..., gt=0)
    tokens_salida:     int       = Field(..., gt=0)
    costo_usd:         float     = Field(..., ge=0.0)
    latencia_ms:       int       = Field(..., gt=0)

    # Regla 3: si la intención es consulta_precio o consulta_stock,
    # producto_detectado no debería estar vacío
    @field_validator("producto_detectado")
    @classmethod
    def producto_presente_si_precio(cls, v: Optional[str], info) -> Optional[str]:
        if v is not None:
            return v.strip() if v.strip() else None
        return v


class ErrorValidacion(BaseModel):
    campo_fallido: str
    detalle:       str
    consulta:      str
    intento:       int  # 1 o 2
