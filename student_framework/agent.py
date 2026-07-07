"""Implementación de su agente.

Completen `register_tool` y `run` para el Milestone 1.
En el Milestone 2 amplíen `MyAgent` para que sea estatal y respete
`max_history_messages`.

Los tests de conformidad en `tests/conformance/test_m1.py` y
`test_m2.py` describen con precisión qué comportamientos deben funcionar
— léanlos antes de empezar.
"""

from __future__ import annotations

import sys
from typing import Any, Callable

from mia_agents.protocols import LLMClient
from mia_agents.types import AgentResult, ToolSchema, AgentStep


class MyAgent:
    def __init__(
        self,
        llm_client: LLMClient,
        system_prompt: str = "Eres un asistente útil.",
        max_iterations: int = 10,
        max_history_messages: int = 50,
        verbose: bool = True,
    ) -> None:
        """Inicializa el agente.

        Parameters
        ----------
        llm_client : LLMClient
            Cliente LLM (real o mock) que el agente utilizará.
        system_prompt : str
            System prompt por defecto.
        max_iterations : int
            Tope de iteraciones del bucle del agente (M1).
        max_history_messages : int
            Número máximo de mensajes que se permiten en la lista
            `messages` enviada al LLM en una única llamada. En M1 este
            valor es ignorado; el agente sólo necesita aceptarlo en su
            constructor. En M2 deben respetarlo: la longitud de la
            lista de mensajes pasada a `self._llm.chat(...)` no puede
            superar este número en ninguna llamada, sin importar la
            estrategia de memoria que elijan.
        """
        self._llm = llm_client
        self._system = system_prompt
        self._max_iterations = max_iterations
        self._max_history_messages = max_history_messages
        self._verbose = verbose
        if verbose:
            # En Windows la consola suele ser cp1252 y los emojis del log
            # romperían el print. Forzamos UTF-8 con reemplazo seguro.
            try:
                sys.stderr.reconfigure(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001 — si el stream no lo soporta, seguimos.
                pass
        # TODO (M1): inicializa el estado interno para las herramientas registradas.
        self._tools: dict[str, Callable[..., str]] = {}
        self._schemas: dict[str, ToolSchema] = {}
        # TODO (M2): inicializa la estructura de historial conversacional.

    def _log(self, message: str) -> None:
        """Imprime trazas del agente a stderr.

        Usamos stderr (no stdout) para no ensuciar el JSON que la CLI
        imprime en stdout. Solo emite si `verbose` está activo.
        """
        if self._verbose:
            print(message, file=sys.stderr, flush=True)

    @staticmethod
    def _short(text: Any, limit: int = 300) -> str:
        """Acorta textos largos (p. ej. el contenido de un archivo) para el log."""
        text = str(text)
        if len(text) <= limit:
            return text
        return text[:limit] + f"... [+{len(text) - limit} caracteres]"

    def register_tool(
        self,
        tool: Callable[..., str],
        schema: ToolSchema,
    ) -> None:
        """Registra una herramienta callable junto a su esquema.

        El esquema suele obtenerse con `ToolSchema.from_callable(fn)`. En
        `run`, pasá `tools=list(self._schemas.values())`; el cliente LLM
        aplica `to_llm_spec()` al llamar al proveedor.

        El callable se invoca con kwargs que coinciden con la firma.
        Debe devolver una cadena.
        """
        self._tools[schema.name] = tool
        self._schemas[schema.name] = schema
        # raise NotImplementedError("M1: implementa el registro de herramientas")
        

    def run(self, user_message: str) -> AgentResult:
        """Ejecuta el bucle del agente hasta una respuesta final o hasta max_iterations.

        Comportamiento esperado (consulta tests/conformance/test_m1.py
        para el contrato exacto del M1):
          - Llama a `self._llm.chat(..., tools=list(self._schemas.values()))`.
          - Si la respuesta contiene tool_calls, ejecuta cada uno y vuelca
            los resultados en la siguiente llamada al chat.
          - Si la respuesta solo contiene texto (sin `tool_calls`),
            devuélvelo en `AgentResult.answer`. En M1 no uses la tool
            sintética `final_result`; ese patrón es de M2 (ver README y
            ENUNCIADO_M2.md).
          - Limita el bucle a `self._max_iterations` y termina de forma
            limpia cuando se alcance.
          - Registra cada invocación de herramienta como un `AgentStep`
            dentro de `result.steps`.

        En el M2, además, llamadas sucesivas sobre la misma instancia
        deben continuar la conversación, y la longitud de la lista de
        mensajes enviada al LLM no debe superar `self._max_history_messages`.
        Acumula los tokens de entrada/salida reportados por los
        `LLMResponse` y exponlos en `AgentResult.input_tokens` /
        `AgentResult.output_tokens`.
        """
        import json

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_message}
        ]
        steps: list[AgentStep] = []

        # tools: nunca None si hay herramientas registradas (lo exige el contrato).
        tools = list(self._schemas.values()) if self._schemas else None

        self._log(f"\n🚀 Usuario: {user_message}")

        last_content = ""
        for iteration in range(1, self._max_iterations + 1):
            self._log(
                f"\n── Iteración {iteration}/{self._max_iterations} · consultando al LLM ──"
            )
            resp = self._llm.chat(
                messages=messages,
                tools=tools,
                system=self._system,
            )

            # Caso 1: el LLM respondió texto sin pedir herramientas -> fin.
            if not resp.tool_calls:
                self._log(f"✅ Respuesta final: {resp.content or ''}")
                return AgentResult(answer=resp.content or "", steps=steps)

            last_content = resp.content or ""

            # Caso 2: el LLM pidió una o más herramientas.
            if resp.content:
                self._log(f"🧠 Razonamiento: {resp.content}")
            self._log(
                f"🔧 Acción: el LLM pide {len(resp.tool_calls)} herramienta(s): "
                + ", ".join(tc.name for tc in resp.tool_calls)
            )

            # Registramos el turno del asistente con sus tool_calls.
            messages.append(
                {
                    "role": "assistant",
                    "content": resp.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "function": {"name": tc.name, "arguments": tc.arguments},
                        }
                        for tc in resp.tool_calls
                    ],
                }
            )

            # Ejecutamos cada herramienta pedida.
            for tc in resp.tool_calls:
                try:
                    kwargs = json.loads(tc.arguments) if tc.arguments else {}
                except json.JSONDecodeError:
                    kwargs = {}

                self._log(f"   ⚙️  Ejecutando {tc.name}({tc.arguments})")

                tool = self._tools.get(tc.name)
                if tool is None:
                    # Herramienta alucinada: no rompemos, registramos el error.
                    error_msg = f"Herramienta desconocida: '{tc.name}'."
                    steps.append(
                        AgentStep(
                            tool_name=tc.name,
                            tool_input=tc.arguments,
                            tool_output=None,
                            error=error_msg,
                        )
                    )
                    output = error_msg
                else:
                    try:
                        output = tool(**kwargs)
                        steps.append(
                            AgentStep(
                                tool_name=tc.name,
                                tool_input=tc.arguments,
                                tool_output=output,
                                error=None,
                            )
                        )
                    except Exception as exc:  # noqa: BLE001
                        output = f"Error ejecutando '{tc.name}': {exc}"
                        steps.append(
                            AgentStep(
                                tool_name=tc.name,
                                tool_input=tc.arguments,
                                tool_output=None,
                                error=str(exc),
                            )
                        )

                self._log(f"   📥 Resultado: {self._short(output)}")

                # Devolvemos el resultado al LLM en la próxima llamada.
                messages.append(
                    {
                        "role": "tool",
                        "content": output,
                        "tool_call_id": tc.id,
                    }
                )

        # Se alcanzó max_iterations: devolvemos un AgentResult válido igual.
        self._log("⚠️  Se alcanzó max_iterations sin respuesta final.")
        return AgentResult(answer=last_content, steps=steps)


    def structured_call(
        self,
        prompt: str,
        schema: Any,
        max_repair_attempts: int = 2,
    ) -> Any:
        """Pide al LLM una respuesta validada contra `schema` (M2).

        Obligatorio: herramienta sintética `final_result` (ver
        `mia_agents.final_result_tool_schema` / `FINAL_RESULT_TOOL_NAME`).
        El agente ofrece esa tool al LLM, valida los `arguments` del
        `tool_call` y reintenta con contexto de reparación si el modelo
        responde con texto libre o con argumentos inválidos.

        Implementa esto en el M2:
          - Pasa `tools=[final_result_tool_schema(schema)]` en cada
            llamada a `chat` dentro de este método.
          - Termina solo cuando llega un `tool_call` a `final_result`
            cuyos argumentos validan con `schema.model_validate(...)`.
          - Reintenta hasta `max_repair_attempts` incluyendo el fallo en
            los mensajes (respuesta previa, mensaje `tool`, o user de
            reparación).
          - Si tras los reintentos sigue fallando, levanta una excepción
            limpia (no devuelvas valores parciales ni `None` sin avisar).

        El M1 deja esto como stub; los tests de M2 verifican el contrato.
        """
        raise NotImplementedError("M2: implementa salida estructurada con reparación")
