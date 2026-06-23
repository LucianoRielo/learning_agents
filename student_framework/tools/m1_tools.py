"""
Herramientas obligatorias del M1:
- Calculadora: `calculate`
- Lector de archivos: `read_file`
- Conversor de unidades: `convert_units`
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from mia_agents.types import ToolSchema

## 1. Calculadora simple
def calculator(
    left_operand: Annotated[float, Field(description="Primer operando numérico.")],
    operator: Annotated[str, Field(description="Operador a aplicar: '+', '-', '*' o '%'.")],
    right_operand: Annotated[float, Field(description="Segundo operando numérico.")],
) -> str:
    """Realiza una operación aritmética binaria entre dos números.

    Operadores soportados: suma (+), resta (-), multiplicación (*) y
    módulo (%). Devuelve el resultado como texto.
    """
    operations = {
        "+": lambda a, b: a + b,
        "-": lambda a, b: a - b,
        "*": lambda a, b: a * b,
        "%": lambda a, b: a % b,
    }
    if operator not in operations:
        return f"Error: operador no soportado '{operator}'. Usá uno de: + - * %"
    if operator == "%" and right_operand == 0:
        return "Error: módulo por cero."
    return str(operations[operator](left_operand, right_operand))

# 2. Lector de archivos
def file_reader(
    path: Annotated[str, Field(description="Ruta al archivo de texto a leer.")],
) -> str:
    """Lee un archivo de texto (UTF-8) y devuelve su contenido completo."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: no se encontró el archivo '{path}'."
    except (OSError, UnicodeDecodeError) as exc:
        return f"Error al leer '{path}': {exc}"
    

# 3. Conversor de unidades
def unit_converter(
    value: Annotated[float, Field(description="Valor numérico a convertir.")],
    from_unit: Annotated[str, Field(description="Unidad de origen: 'km', 'mi', 'c' o 'f'.")],
    to_unit: Annotated[str, Field(description="Unidad de destino: 'km', 'mi', 'c' o 'f'.")],
) -> str:
    """Convierte un valor entre unidades.

    Soporta distancia (kilómetros 'km' ↔ millas 'mi') y temperatura
    (Celsius 'c' ↔ Fahrenheit 'f'). Devuelve el valor convertido como texto.
    """
    conversions = {
        ("km", "mi"): lambda v: v * 0.621371,
        ("mi", "km"): lambda v: v / 0.621371,
        ("c", "f"): lambda v: v * 9 / 5 + 32,
        ("f", "c"): lambda v: (v - 32) * 5 / 9,
    }
    key = (from_unit.lower(), to_unit.lower())
    if from_unit.lower() == to_unit.lower():
        return str(value)
    if key not in conversions:
        return f"Error: no sé convertir de '{from_unit}' a '{to_unit}'."
    return str(conversions[key](value))


# Esquemas para el LLM (NO escribir JSON Schema a mano: from_callable lo deriva)
calculator_schema = ToolSchema.from_callable(calculator)
file_reader_schema = ToolSchema.from_callable(file_reader)
unit_converter_schema = ToolSchema.from_callable(unit_converter)