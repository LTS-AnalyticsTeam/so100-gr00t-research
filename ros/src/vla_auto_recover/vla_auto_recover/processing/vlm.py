import base64
import cv2
import os
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
from .state import State


class VisionAnalyzer:
    """Visual Language Model for anomaly detection and action suggestion"""

    MODEL = "gpt-4.1"

    def __init__(self):
        config_path = Path(__file__).parent.joinpath("config")
        self.client, self.use_azure = self._init_openai_client()
        with open(config_path.joinpath("anomaly_detection_prompt.txt"), "r") as f:
            self.anomaly_detection_prompt = f.read().strip()
        with open(config_path.joinpath("anomaly_detection_prompt.txt"), "r") as f:
            self.anomaly_detection_prompt = f.read().strip()
        with open(config_path.joinpath("action_list.jsonl"), "r") as f:
            self.actions_txt = f.read().strip()

    def _init_openai_client(self):
        """Initialize OpenAI client (Azure or OpenAI)"""

        env_file = Path(__file__).parent.joinpath("config", ".env")
        load_dotenv(env_file)

        use_azure = os.getenv("USE_AZURE_OPENAI", "false").lower() == "true"

        try:
            if use_azure:
                from openai import AzureOpenAI

                client = AzureOpenAI(
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    api_version=os.getenv(
                        "AZURE_OPENAI_API_VERSION", "2024-02-15-preview"
                    ),
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

    def detect_anomaly(self, images: list[np.ndarray]) -> tuple[State, str]:
        """Detect anomalies and suggest actions using VLM"""
        openai_images = [self._transform_image_for_openai(img) for img in images]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt},
                    {"type": "text", "text": self.actions_txt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                ],
            }
        ]

        response = self.client.chat.completions.create(
            model=self.MODEL, messages=messages
        )
        return State.NORMAL, response.choices[0].message.content

    @staticmethod
    def _transform_image_for_openai(self, image: np.ndarray) -> str:
        """Convert OpenCV image to base64 format for OpenAI API"""
        _, buffer = cv2.imencode(".jpg", image)
        base64_image = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_image}"
