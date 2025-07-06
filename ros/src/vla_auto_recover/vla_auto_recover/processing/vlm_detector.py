import base64
import cv2
import os
import json
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
from .config import prompt_settings as ps
from .config.system_settings import ADR, RDR, VDR, State
from .config.system_settings import CB_InputIF, CB_OutputIF, CB_PREFIX

ENV_FILE = "/workspace/ros/src/vla_auto_recover/config/.env"

class BaseDetector:

    def call_CB(self, state: State, input_data: CB_InputIF) -> CB_OutputIF:
        # Callbackの実行
        if state.value == "RUNNING":
            output_data = self.CB_RUNNING(input_data)
        elif state.value == "RECOVERY":
            output_data = self.CB_RECOVERY(input_data)
        elif state.value == "VERIFICATION":
            output_data = self.CB_VERIFICATION(input_data)
        elif state.value == "END":
            output_data = self.CB_END(input_data)
        else:
            raise ValueError(f"Unknown state: {state.value}")
        return output_data

    def CB_RUNNING(self, input_data: CB_InputIF) -> CB_OutputIF:
        print("call: CB_RUNNING")
        return CB_OutputIF()

    def CB_RECOVERY(self, input_data: CB_InputIF) -> CB_OutputIF:
        print("call: CB_RECOVERY")
        return CB_OutputIF()

    def CB_VERIFICATION(self, input_data: CB_InputIF) -> CB_OutputIF:
        print("call: CB_VERIFICATION")
        return CB_OutputIF()

    def CB_END(self, input_data: CB_InputIF) -> CB_OutputIF:
        print("call: CB_END")
        return CB_OutputIF()


class VLMDetector(BaseDetector):
    """Visual Language Model for anomaly detection and action suggestion"""

    MODEL = "gpt-4.1"

    def __init__(self):
        self.client, self.use_azure = self._init_openai_client()

    def _init_openai_client(self):
        """Initialize OpenAI client (Azure or OpenAI)"""

        load_dotenv(ENV_FILE)

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

    def CB_RUNNING(self, input_data: CB_InputIF) -> CB_OutputIF:
        """Detect anomalies and suggest actions using VLM"""
        openai_images = [
            self._transform_image_for_openai(img) for img in input_data.images
        ]

        # messagesの作成
        content = []
        prompt = ps.CB_RUNNING_PROMPT.format(
            language_instruction=ps.RUNNING_LANGUAGE_INSTRUCTION,
            recovery_action_list=json.dumps(
                ps.RECOVERY_ACTION_LIST, indent=2, ensure_ascii=False
            ),
        )
        content.append({"type": "text", "text": prompt})
        for img in openai_images:
            content.append({"type": "image_url", "image_url": {"url": img}})

        print(f"prompt: \n{prompt}")
        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[{"role": "user", "content": content}],
            response_format={
                "type": "json_schema",
                "json_schema": ps.CB_RUNNING_JSON_SCHEMA,
            },
        )
        output_data = self._parse_CB_RUNNING_response(response)
        return output_data

    def CB_RECOVERY(self, input_data: CB_InputIF) -> CB_OutputIF:
        """Check if the system is recovering from an anomaly"""
        openai_images = [
            self._transform_image_for_openai(img) for img in input_data.images
        ]

        # Get action instruction based on action_id
        language_instruction = ps.RECOVERY_ACTION_LIST[input_data.action_id][
            "language_instruction"
        ]

        # messagesの作成
        content = []
        prompt = ps.CB_RECOVERY_PROMPT.format(
            language_instruction=language_instruction,
        )
        content.append({"type": "text", "text": prompt})
        for img in openai_images:
            content.append({"type": "image_url", "image_url": {"url": img}})

        print(f"prompt: \n{prompt}")
        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[{"role": "user", "content": content}],
            response_format={
                "type": "json_schema",
                "json_schema": ps.CB_RECOVERY_JSON_SCHEMA,
            },
        )
        output_data = self._parse_CB_RECOVERY_response(response)
        # CB_RECOVERYではアクションを変えない。
        output_data.action_id = input_data.action_id
        return output_data

    def CB_VERIFICATION(self, input_data: CB_InputIF) -> CB_OutputIF:
        """Verify if the anomaly has been resolved"""
        openai_images = [
            self._transform_image_for_openai(img) for img in input_data.images
        ]

        # messagesの作成
        content = []
        prompt = ps.CB_VERIFICATION_PROMPT.format(
            recovery_action_list=json.dumps(
                ps.RECOVERY_ACTION_LIST, indent=2, ensure_ascii=False
            ),
        )
        content.append({"type": "text", "text": prompt})
        for img in openai_images:
            content.append({"type": "image_url", "image_url": {"url": img}})

        print(f"prompt: \n{prompt}")
        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[{"role": "user", "content": content}],
            response_format={
                "type": "json_schema",
                "json_schema": ps.CB_VERIFICATION_JSON_SCHEMA,
            },
        )
        output_data = self._parse_CB_VERIFICATION_response(response)
        return output_data

    def CB_END(self, input_data: CB_InputIF) -> CB_OutputIF:
        return CB_OutputIF()

    def _parse_CB_RUNNING_response(self, response) -> CB_OutputIF:
        """Parse and validate anomaly detection response"""
        try:
            response_text = response.choices[0].message.content
            parsed_response = json.loads(response_text)

            # JSON schema validation
            if "detection_result" not in parsed_response:
                raise ValueError(
                    "Missing required field 'detection_result' in response"
                )
            if "action_id" not in parsed_response:
                raise ValueError("Missing required field 'action_id' in response")
            if "reason" not in parsed_response:
                raise ValueError("Missing required field 'reason' in response")

            detection_result_str = parsed_response["detection_result"]
            action_id = parsed_response["action_id"]
            reason = parsed_response["reason"]

            # Validate state value
            if detection_result_str not in ["NORMAL", "ANOMALY", "COMPLETION"]:
                raise ValueError(
                    f"Invalid state value: {detection_result_str}. Expected 'NORMAL', 'ANOMALY', or 'COMPLETION'"
                )

            # Convert string to State enum
            return CB_OutputIF(
                detection_result=ADR(detection_result_str),
                action_id=action_id,
                reason=reason,
            )

        except (json.JSONDecodeError, KeyError, AttributeError, ValueError) as e:
            print(f"Error parsing anomaly detection response: {e}")
            print(
                f"Raw response: {response.choices[0].message.content if response.choices else 'No response'}"
            )
            return CB_OutputIF(detection_result=None, action_id=None, reason=None)

    def _parse_CB_RECOVERY_response(self, response) -> CB_OutputIF:
        """Parse and validate recovery state response"""
        try:
            response_text = response.choices[0].message.content
            parsed_response = json.loads(response_text)

            # JSON schema validation
            if "detection_result" not in parsed_response:
                raise ValueError(
                    "Missing required field 'detection_result' in response"
                )
            if "reason" not in parsed_response:
                raise ValueError("Missing required field 'reason' in response")

            detection_result_str = parsed_response["detection_result"]
            reason = parsed_response["reason"]

            # Validate detection_result value
            if detection_result_str not in ["RECOVERED", "UNRECOVERED"]:
                raise ValueError(
                    f"Invalid detection_result value: {detection_result_str}. Expected 'RECOVERED' or 'UNRECOVERED'"
                )

            # Convert string to RDR enum
            return CB_OutputIF(
                detection_result=RDR(detection_result_str),
                action_id=None,
                reason=reason,
            )

        except (json.JSONDecodeError, KeyError, AttributeError, ValueError) as e:
            print(f"Error parsing recovery state response: {e}")
            print(
                f"Raw response: {response.choices[0].message.content if response.choices else 'No response'}"
            )
            return CB_OutputIF(detection_result=None, action_id=None, reason=None)

    def _parse_CB_VERIFICATION_response(self, response) -> CB_OutputIF:
        """Parse and validate verification state response"""
        try:
            response_text = response.choices[0].message.content
            parsed_response = json.loads(response_text)

            # JSON schema validation
            if "detection_result" not in parsed_response:
                raise ValueError(
                    "Missing required field 'detection_result' in response"
                )
            if "action_id" not in parsed_response:
                raise ValueError("Missing required field 'action_id' in response")
            if "reason" not in parsed_response:
                raise ValueError("Missing required field 'reason' in response")

            detection_result_str = parsed_response["detection_result"]
            action_id = parsed_response["action_id"]
            reason = parsed_response["reason"]

            # Validate detection_result value
            if detection_result_str not in ["SOLVED", "UNSOLVED"]:
                raise ValueError(
                    f"Invalid detection_result value: {detection_result_str}. Expected 'SOLVED' or 'UNSOLVED'"
                )

            # Convert string to VDR enum
            return CB_OutputIF(
                detection_result=VDR(detection_result_str),
                action_id=action_id,
                reason=reason,
            )

        except (json.JSONDecodeError, KeyError, AttributeError, ValueError) as e:
            print(f"Error parsing verification state response: {e}")
            print(
                f"Raw response: {response.choices[0].message.content if response.choices else 'No response'}"
            )
            return CB_OutputIF(detection_result=None, action_id=None, reason=None)

    def _transform_image_for_openai(self, image: np.ndarray) -> str:
        """Convert OpenCV image to base64 format for OpenAI API"""
        _, buffer = cv2.imencode(".jpg", image)
        base64_image = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_image}"
