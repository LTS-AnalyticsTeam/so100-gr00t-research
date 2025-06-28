import os
from pathlib import Path
from dotenv import load_dotenv
from .state import State


def init_openai_client():
    """Initialize OpenAI client (Azure or OpenAI)"""

    env_file = Path(__file__).parent.joinpath("config", ".env")
    load_dotenv(env_file)

    use_azure = os.getenv("USE_AZURE_OPENAI", "false").lower() == "true"

    try:
        if use_azure:
            from openai import AzureOpenAI

            client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            )
            print("Azure OpenAI client initialized")
        else:
            from openai import OpenAI

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            print("OpenAI client initialized")

        return client, use_azure

    except Exception as e:
        print(f"Warning: OpenAI client not available: {e}")
        return None, use_azure


# Initialize OpenAI client
CLIENT, use_azure = init_openai_client()


def detect_anomaly_and_suggest_action(
    images, language_instruction: str
) -> tuple[State, str]:
    """Detect anomalies and suggest actions using VLM"""
