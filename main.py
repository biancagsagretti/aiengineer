import os
import asyncio
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import SecretStr, ValidationError

from schemas import ChatMessage, LLMConfig, Provider
from manager import AsyncLLMManager

load_dotenv()

PREGUNTA: List[ChatMessage] = [
    ChatMessage(role="system", content="Sos un asistente especialista en derecho civil argentino."),
    ChatMessage(role="user", content="¿Qué es un acto jurídico?"),
]


def probar_validacion() -> None:
    """Demuestra que Pydantic detecta parámetros inválidos ANTES de llamar a la API."""
    print("🧪 Probando validación de Pydantic (temperature=5)...")
    try:
        LLMConfig(provider=Provider.OPENAI, model="gpt-4o-mini", temperature=5)
    except ValidationError as e:
        print("❌ Se detectó ANTES de llamar a la API:\n", e)


def crear_manager(
    provider: Provider, model: str, env_var: str, campo_key: str
) -> Optional[AsyncLLMManager]:
    """Crea un manager si la API key existe en el entorno; si no, devuelve None."""
    api_key = os.getenv(env_var)
    if not api_key:
        print(f"⏭️  {provider.value} omitido: falta {env_var} en el .env")
        return None
    config = LLMConfig(
        provider=provider,
        model=model,
        temperature=0.7,
        max_tokens=300,
        **{campo_key: SecretStr(api_key)},
    )
    return AsyncLLMManager(config)


async def probar_proveedor(nombre: str, manager: AsyncLLMManager) -> None:
    """Prueba el modo normal y el modo streaming de un proveedor."""
    print(f"\n=== {nombre} ===")

    print("▶ Modo normal:")
    resultado = await manager.generate(PREGUNTA)
    print(resultado.content if not resultado.error else f"❌ {resultado.error}")

    print("\n▶ Modo streaming:")
    async for chunk in manager.generate_stream(PREGUNTA):
        print(chunk, end="", flush=True)
    print()


async def probar_resiliencia() -> None:
    """Una API key inválida NO debe romper el programa: debe devolver un error controlado."""
    print("\n🛠️  Probando resiliencia con una API key inválida...")
    config_rota = LLMConfig(
        provider=Provider.OPENAI,
        model="gpt-4o-mini",
        openai_api_key=SecretStr("sk-key-invalida-a-proposito"),
    )
    manager_roto = AsyncLLMManager(config_rota)
    resultado = await manager_roto.generate(PREGUNTA)

    print("¿El programa siguió vivo?: ✅ Sí")
    print("Error capturado (sin crash):", resultado.error)


async def main() -> None:
    probar_validacion()

    proveedores = [
        ("🟢 OpenAI", crear_manager(
            Provider.OPENAI, "gpt-4o-mini", "OPENAI_API_KEY", "openai_api_key")),
        ("🟣 Anthropic", crear_manager(
            Provider.ANTHROPIC, "claude-haiku-4-5-20251001", "ANTHROPIC_API_KEY", "anthropic_api_key")),
        ("🔵 Gemini", crear_manager(
            Provider.GEMINI, "gemini-flash-latest", "GOOGLE_API_KEY", "google_api_key")),
    ]

    for nombre, manager in proveedores:
        if manager is not None:
            await probar_proveedor(nombre, manager)

    await probar_resiliencia()


if __name__ == "__main__":
    asyncio.run(main())
