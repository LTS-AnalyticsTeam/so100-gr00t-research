import base64
import cv2
import os
import json
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
from enum import Enum
from .config import settings
from .config.settings import State


class VLMMonitor:
    """Visual Language Model for anomaly detection and action suggestion"""

    MODEL = "gpt-4.1"

    def __init__(self):
        self.client, self.use_azure = self._init_openai_client()

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

    def detect_anomaly(
        self, images: list[np.ndarray], action_instruction: str
    ) -> tuple[State, int]:
        """Detect anomalies and suggest actions using VLM"""
        openai_images = [self._transform_image_for_openai(img) for img in images]

        # messagesの作成
        content = []
        prompt = settings.ANOMALY_DETECTION_PROMPT.format(
            action_instruction=action_instruction,
            action_list=json.dumps(settings.ACTION_LIST, indent=2, ensure_ascii=False),
        )
        content.append({"type": "text", "text": prompt})
        for img in openai_images:
            content.append({"type": "image_url", "image_url": {"url": img}})

        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[{"role": "user", "content": content}],
            response_format={
                "type": "json_schema",
                "json_schema": settings.JSON_SCHEMA_ANOMALY_DETECTION,
            },
        )

        return self._parse_anomaly_detection_response(response)

    def check_recovery_state(
        self, images: list[np.ndarray], action_instruction: str
    ) -> State:
        """Check if the system is recovering from an anomaly"""
        openai_images = [self._transform_image_for_openai(img) for img in images]

        # messagesの作成
        content = []
        prompt = settings.RECOVERY_STATE_PROMPT.format(
            action_instruction=action_instruction
        )
        content.append({"type": "text", "text": prompt})
        for img in openai_images:
            content.append({"type": "image_url", "image_url": {"url": img}})

        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[{"role": "user", "content": content}],
            response_format={
                "type": "json_schema",
                "json_schema": settings.JSON_SCHEMA_RECOVERY_STATE,
            },
        )

        return self._parse_recovery_state_response(response)

    def _parse_anomaly_detection_response(self, response) -> tuple[State, int]:
        """Parse and validate anomaly detection response"""
        try:
            response_text = response.choices[0].message.content
            parsed_response = json.loads(response_text)

            # JSON schema validation
            if "state" not in parsed_response:
                raise ValueError("Missing required field 'state' in response")
            if "action_id" not in parsed_response:
                raise ValueError("Missing required field 'action_id' in response")

            state_str = parsed_response["state"]
            action_id = parsed_response["action_id"]
            reason = parsed_response["reason"]

            # Validate state value
            if state_str not in ["NORMAL", "ANOMALY"]:
                raise ValueError(
                    f"Invalid state value: {state_str}. Expected 'NORMAL' or 'ANOMALY'"
                )

            # Convert string to State enum
            return State(state_str), action_id, reason

        except (json.JSONDecodeError, KeyError, AttributeError, ValueError) as e:
            print(f"Error parsing anomaly detection response: {e}")
            print(
                f"Raw response: {response.choices[0].message.content if response.choices else 'No response'}"
            )
            return None, None, None

    def _parse_recovery_state_response(self, response) -> State:
        """Parse and validate recovery state response"""
        try:
            response_text = response.choices[0].message.content
            parsed_response = json.loads(response_text)

            if "state" not in parsed_response:
                raise ValueError("Missing required field 'state' in response")

            state_str = parsed_response["state"]
            reason = parsed_response["reason"]

            # Validate state value
            if state_str not in ["NORMAL", "ANOMALY"]:
                raise ValueError(
                    f"Invalid state value: {state_str}. Expected 'NORMAL' or 'ANOMALY'"
                )

            return State(state_str), reason

        except (json.JSONDecodeError, KeyError, AttributeError, ValueError) as e:
            print(f"Error parsing recovery state response: {e}")
            print(
                f"Raw response: {response.choices[0].message.content if response.choices else 'No response'}"
            )
            return None, None

    def _transform_image_for_openai(self, image: np.ndarray) -> str:
        """Convert OpenCV image to base64 format for OpenAI API"""
        _, buffer = cv2.imencode(".jpg", image)
        base64_image = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_image}"
