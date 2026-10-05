# Unified Async LLM Client

Cliente asíncrono y unificado para modelos de lenguaje. Permite usar **OpenAI**, **Anthropic** y **Gemini** bajo una misma interfaz, con soporte para respuestas completas y **streaming**, validación de datos con **Pydantic** y manejo controlado de errores (sin crashes ante API keys inválidas o límites de cuota).

## Características

- **Intercambiabilidad:** una clase base abstracta (`BaseLLMClient`) y un `AsyncLLMManager` que elige el proveedor según la configuración.
- **Asincronía:** todas las llamadas usan `async/await` con los SDKs asíncronos (`AsyncOpenAI`, `AsyncAnthropic` y `client.aio` de Gemini), sin bloquear el event loop.
- **Streaming:** `generate_stream()` es un generador asíncrono que entrega fragmentos de texto con `yield` a medida que llegan de la API.
- **Validación:** esquemas Pydantic para mensajes (`ChatMessage`), configuración (`LLMConfig`) y respuestas (`ModelResponse`). La temperatura se valida entre 0 y 2, y `max_tokens` debe ser mayor que 0.
- **Errores controlados:** los errores de red, de cuota (rate limit) y de API se devuelven como un `ModelResponse` con el campo `error`, en lugar de lanzar una excepción.
- **Seguridad:** las API keys se manejan con `SecretStr` para que no se impriman por accidente.

## Estructura del proyecto

```
.
├── schemas.py        # Provider, ChatMessage, LLMConfig, ModelResponse (Pydantic)
├── clients.py        # BaseLLMClient, OpenAIClient, AnthropicClient, GeminiClient
├── manager.py        # AsyncLLMManager (elige el cliente según la config)
├── main.py           # Script de validación: modo normal, streaming y resiliencia
├── requirements.txt  # Dependencias
├── .env.example      # Plantilla de variables de entorno
├── .gitignore        # Excluye .env del repositorio
└── README.md
```

## Requisitos

- Python **3.12**
- Una API key de al menos uno de los proveedores (OpenAI, Anthropic o Google Gemini)

## Instalación

1. **Clonar el repositorio:**

   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd <NOMBRE_DEL_REPOSITORIO>
   ```

2. **Crear y activar un entorno virtual con Python 3.12:**

   ```bash
   python3.12 -m venv .venv

   # Linux / macOS
   source .venv/bin/activate

   # Windows (PowerShell)
   .venv\Scripts\Activate.ps1
   ```

3. **Instalar las dependencias:**

   ```bash
   pip install -r requirements.txt
   ```

   Las dependencias son `openai`, `anthropic`, `google-genai`, `pydantic` y `python-dotenv`.

## Variables de entorno

Copiá la plantilla y completá tus claves:

```bash
cp .env.example .env
```

| Variable            | Descripción                          | Obligatoria si usás |
|---------------------|--------------------------------------|---------------------|
| `OPENAI_API_KEY`    | API key de OpenAI                    | OpenAI              |
| `ANTHROPIC_API_KEY` | API key de Anthropic                 | Anthropic           |
| `GOOGLE_API_KEY`    | API key de Google AI Studio (Gemini) | Gemini              |



## Cómo ejecutar el script de prueba

```bash
python main.py
```

El script `main.py` carga el `.env` y realiza lo siguiente:

1. **Validación Pydantic:** intenta crear una configuración con `temperature=5` y muestra que el error se detecta *antes* de llamar a la API.
2. **Modo normal:** envía una pregunta corta a cada proveedor y muestra la respuesta completa.
3. **Modo streaming:** envía la misma pregunta e imprime los fragmentos de texto a medida que llegan.
4. **Prueba de resiliencia:** usa una API key inválida a propósito y verifica que el programa no se rompe, sino que devuelve un error controlado.

## Uso en tu propio código

```python
import asyncio
from pydantic import SecretStr

from schemas import ChatMessage, LLMConfig, Provider
from manager import AsyncLLMManager


async def demo():
    config = LLMConfig(
        provider=Provider.OPENAI,          # o Provider.ANTHROPIC / Provider.GEMINI
        model="gpt-4o-mini",
        openai_api_key=SecretStr("tu_api_key"),
        temperature=0.7,
        max_tokens=300,
    )
    manager = AsyncLLMManager(config)

    mensajes = [
        ChatMessage(role="system", content="Sos un asistente legal especializado en derecho civil argentino."),
        ChatMessage(role="user", content="¿Qué es un acto jurídico?"),
    ]

    # Modo normal
    respuesta = await manager.generate(mensajes)
    print(respuesta.content if not respuesta.error else f"Error: {respuesta.error}")

    # Modo streaming
    async for fragmento in manager.generate_stream(mensajes):
        print(fragmento, end="", flush=True)


asyncio.run(demo())
```

Para cambiar de proveedor alcanza con modificar `provider`, `model` y la API key correspondiente en `LLMConfig`. El resto del código no cambia.

## Manejo de errores

| Situación                          | Comportamiento                                                                 |
|------------------------------------|--------------------------------------------------------------------------------|
| Parámetros inválidos (temperatura fuera de 0–2, rol desconocido, etc.) | `ValidationError` de Pydantic antes de llamar a la API |
| Falta la API key en la configuración | `ValueError` al crear el `AsyncLLMManager`                                   |
| Límite de cuota, error de red o de API (modo normal) | `ModelResponse` con `content=""` y el detalle en `error` |
| Error durante el streaming         | Se emite un mensaje de error en el flujo y el programa continúa                |

## Notas

- **Modelos:** los nombres de modelo cambian con frecuencia. Si recibís un error de "modelo no encontrado", consultá la documentación del proveedor y actualizá el valor de `model` en `main.py`.
- **System prompt:** OpenAI acepta el rol `system` dentro de la lista de mensajes, mientras que Anthropic y Gemini lo reciben como un parámetro aparte. Los clientes se encargan de esa conversión internamente.
