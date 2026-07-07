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
def calculate(
    left_operand: Annotated[float, Field(description="Primer operando numérico.")],
    operator: Annotated[str, Field(description="Operador a aplicar: '+', '-', '*' o '%'.")],
    right_operand: Annotated[float, Field(description="Segundo operando numérico.")],
) -> str:
    """Realiza una operación aritmética binaria entre dos números.

    Operadores soportados: suma (+), resta (-), multiplicación (*) y
    módulo (%). Devuelve el resultado como texto.
    """
    # Los LLMs suelen mandar los números como texto ("2" en vez de 2).
    # Coaccionamos a float antes de operar para no concatenar ni crashear.
    try:
        a = float(left_operand)
        b = float(right_operand)
    except (TypeError, ValueError):
        return f"Error: operandos no numéricos ('{left_operand}', '{right_operand}')."

    operations = {
        "+": lambda x, y: x + y,
        "-": lambda x, y: x - y,
        "*": lambda x, y: x * y,
        "%": lambda x, y: x % y,
    }
    if operator not in operations:
        return f"Error: operador no soportado '{operator}'. Usá uno de: + - * %"
    if operator == "%" and b == 0:
        return "Error: módulo por cero."
    return str(operations[operator](a, b))

# 2. Lector de archivos
def read_file(
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
def convert_units(
    value: Annotated[float, Field(description="Valor numérico a convertir.")],
    from_unit: Annotated[str, Field(description="Unidad de origen: 'km', 'mi', 'c' o 'f'.")],
    to_unit: Annotated[str, Field(description="Unidad de destino: 'km', 'mi', 'c' o 'f'.")],
) -> str:
    """Convierte un valor entre unidades.

    Soporta distancia (kilómetros 'km' ↔ millas 'mi') y temperatura
    (Celsius 'c' ↔ Fahrenheit 'f'). Devuelve el valor convertido como texto.
    """
    # Igual que en calculate: el LLM puede mandar el número como texto.
    try:
        v = float(value)
    except (TypeError, ValueError):
        return f"Error: 'value' no es numérico ('{value}')."

    conversions = {
        ("km", "mi"): lambda x: x * 0.621371,
        ("mi", "km"): lambda x: x / 0.621371,
        ("c", "f"): lambda x: x * 9 / 5 + 32,
        ("f", "c"): lambda x: (x - 32) * 5 / 9,
    }
    key = (from_unit.lower(), to_unit.lower())
    if from_unit.lower() == to_unit.lower():
        return str(v)
    if key not in conversions:
        return f"Error: no sé convertir de '{from_unit}' a '{to_unit}'."
    return str(conversions[key](v))


# Esquemas para el LLM (NO escribir JSON Schema a mano: from_callable lo deriva)
calculate_schema = ToolSchema.from_callable(calculate)
read_file_schema = ToolSchema.from_callable(read_file)
convert_units_schema = ToolSchema.from_callable(convert_units)