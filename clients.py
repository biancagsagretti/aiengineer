from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Optional, Tuple

from openai import AsyncOpenAI, APIError, RateLimitError, APIConnectionError
from anthropic import (
    AsyncAnthropic,
    APIError as AnthropicAPIError,
    RateLimitError as AnthropicRateLimitError,
    APIConnectionError as AnthropicConnectionError,
)
from google import genai
from google.genai import types

from schemas import ChatMessage, ModelResponse, Provider


class BaseLLMClient(ABC):
    """Contrato que todo cliente de LLM debe cumplir, sin importar el proveedor real detrás."""

    @abstractmethod
    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        """Genera una respuesta completa (modo normal, no streaming)."""
        raise NotImplementedError

    @abstractmethod
    async def generate_stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        """Genera la respuesta token a token (modo streaming)."""
        raise NotImplementedError
        yield  # nunca se ejecuta; solo le indica a Python que este método es un generador


# =====================================================================
# OPENAI
# =====================================================================
class OpenAIClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        self._client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _error(self, mensaje: str) -> ModelResponse:
        return ModelResponse(provider=Provider.OPENAI, model=self.model, content="", error=mensaje)

    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[m.model_dump() for m in messages],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return ModelResponse(
                provider=Provider.OPENAI,
                model=self.model,
                content=response.choices[0].message.content or "",
            )
        except RateLimitError as e:
            return self._error(f"Límite de cuota excedido: {e}")
        except APIConnectionError as e:
            return self._error(f"Error de conexión: {e}")
        except APIError as e:
            return self._error(f"Error de la API de OpenAI: {e}")
        except Exception as e:
            return self._error(f"Error inesperado: {e}")

    async def generate_stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        try:
            stream = await self._client.chat.completions.create(
                model=self.model,
                messages=[m.model_dump() for m in messages],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:
            yield f"\n[⚠️ Error durante el streaming: {e}]"


# =====================================================================
# ANTHROPIC
# =====================================================================
class AnthropicClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        self._client = AsyncAnthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _error(self, mensaje: str) -> ModelResponse:
        return ModelResponse(provider=Provider.ANTHROPIC, model=self.model, content="", error=mensaje)

    @staticmethod
    def _separar_system(messages: List[ChatMessage]) -> Tuple[Optional[str], List[dict]]:
        """Anthropic no acepta el rol 'system' dentro de messages: va en el parámetro 'system'."""
        system_partes = [m.content for m in messages if m.role == "system"]
        system = "\n".join(system_partes) if system_partes else None
        resto = [m.model_dump() for m in messages if m.role != "system"]
        return system, resto

    def _kwargs(self, messages: List[ChatMessage]) -> dict:
        system, resto = self._separar_system(messages)
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,  # obligatorio en Anthropic, a diferencia de OpenAI
            "temperature": self.temperature,
            "messages": resto,
        }
        if system:
            kwargs["system"] = system
        return kwargs

    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        try:
            response = await self._client.messages.create(**self._kwargs(messages))
            texto = "".join(b.text for b in response.content if b.type == "text")
            return ModelResponse(provider=Provider.ANTHROPIC, model=self.model, content=texto)
        except AnthropicRateLimitError as e:
            return self._error(f"Límite de cuota excedido: {e}")
        except AnthropicConnectionError as e:
            return self._error(f"Error de conexión: {e}")
        except AnthropicAPIError as e:
            return self._error(f"Error de la API de Anthropic: {e}")
        except Exception as e:
            return self._error(f"Error inesperado: {e}")

    async def generate_stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        try:
            async with self._client.messages.stream(**self._kwargs(messages)) as stream:
                async for texto in stream.text_stream:
                    yield texto
        except Exception as e:
            yield f"\n[⚠️ Error durante el streaming: {e}]"


# =====================================================================
# GEMINI
# =====================================================================
class GeminiClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        self._client = genai.Client(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _convertir_mensajes(self, messages: List[ChatMessage]):
        """Gemini separa el system prompt del resto, y llama 'model' al rol del asistente."""
        contents = []
        system_instruction = None
        for m in messages:
            if m.role == "system":
                system_instruction = m.content
            else:
                rol_gemini = "model" if m.role == "assistant" else "user"
                contents.append(types.Content(role=rol_gemini, parts=[types.Part(text=m.content)]))
        return contents, system_instruction

    def _config(self, system_instruction: Optional[str]) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            system_instruction=system_instruction,
        )

    async def generate(self, messages: List[ChatMessage]) -> ModelResponse:
        try:
            contents, system_instruction = self._convertir_mensajes(messages)
            response = await self._client.aio.models.generate_content(
                model=self.model,
                contents=contents,
                config=self._config(system_instruction),
            )
            return ModelResponse(
                provider=Provider.GEMINI, model=self.model, content=response.text or ""
            )
        except Exception as e:
            return ModelResponse(provider=Provider.GEMINI, model=self.model, content="",
                                  error=f"Error de la API de Gemini: {e}")

    async def generate_stream(self, messages: List[ChatMessage]) -> AsyncGenerator[str, None]:
        try:
            contents, system_instruction = self._convertir_mensajes(messages)
            stream = await self._client.aio.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=self._config(system_instruction),
            )
            async for chunk in stream:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            yield f"\n[⚠️ Error durante el streaming: {e}]"
