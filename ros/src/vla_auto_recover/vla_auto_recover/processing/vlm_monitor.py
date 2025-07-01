import base64
import cv2
import os
import json
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
from .config import prompt_settings as ps
from .config.system_state import ADR, RDR, VDR, State
from enum import Enum
from transitions import Machine
from dataclasses import dataclass


@dataclass
class CB_InputIF:
    images: list[np.ndarray] = None
    action_id: int = None


@dataclass
class CB_OutputIF:
    detection_result: State = None
    action_id: int = None
    reason: str = None


class VLMDetector:
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

    def CB_NORMAL(self, input_data: CB_InputIF) -> CB_OutputIF:
        """Detect anomalies and suggest actions using VLM"""
        openai_images = [
            self._transform_image_for_openai(img) for img in input_data.images
        ]

        # messagesの作成
        content = []
        prompt = ps.CB_NORMAL_PROMPT.format(
            language_instruction=ps.NORMAL_LANGUAGE_INSTRUCTION,
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
                "json_schema": ps.CB_NORMAL_JSON_SCHEMA,
            },
        )

        return self._parse_CB_NORMAL_response(response)

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

        return self._parse_CB_RECOVERY_response(response)

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

        return self._parse_CB_VERIFICATION_response(response)

    def _parse_CB_NORMAL_response(self, response) -> CB_OutputIF:
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


class VLMMonitor:
    state: str
    states = [
        State.NORMAL.value,
        State.RECOVERY.value,
        State.VERIFICATION.value,
        State.COMPLETION.value,
    ]
    transitions = [
        {"trigger": ADR.NORMAL.value, "source": State.NORMAL.value, "dest": State.NORMAL.value},
        {"trigger": ADR.ANOMALY.value, "source": State.NORMAL.value, "dest": State.RECOVERY.value},
        {"trigger": ADR.COMPLETION.value, "source": State.NORMAL.value, "dest": State.COMPLETION.value},
        {"trigger": RDR.UNRECOVERED.value, "source": State.RECOVERY.value, "dest": State.RECOVERY.value},
        {"trigger": RDR.RECOVERED.value, "source": State.RECOVERY.value, "dest": State.VERIFICATION.value},
        {"trigger": VDR.SOLVED.value, "source": State.VERIFICATION.value, "dest": State.NORMAL.value},
        {"trigger": VDR.UNSOLVED.value, "source": State.VERIFICATION.value, "dest": State.RECOVERY.value},
    ] # fmt: skip

    def __init__(self):
        self.machine = Machine(
            model=self,
            states=self.states,
            transitions=self.transitions,
            initial=State.NORMAL.value,
        )
        self.vlm = VLMDetector()

    def call_current_CB(self, CB_Interface: CB_InputIF) -> CB_OutputIF:
        return getattr(self.vlm, self._current_cb_name())(input_data=CB_Interface)

    def write_mermaid(self):
        lines = [
            "```mermaid",
            "---",
            "config:",
            "  theme: redux",
            "  flowchart:",
            "    nodeSpacing: 300",
            "    rankSpacing: 60",
            "    curve: basis",
            "---",
        ]
        lines.append("flowchart TD")
        for state in self.states:
            lines.append(f"    {state}({state})")
            lines.append(f"    {state}({state}) --> {self._get_cb_name(state)}{{{self._get_cb_name(state)}}}") # fmt: skip
        for linkage in self.transitions:
            lines.append(
                f"    {self._get_cb_name(linkage['source'])}{{{self._get_cb_name(linkage['source'])}}} -- {linkage['trigger']} --> {linkage['dest']}"
            )
        lines.append("```")
        return "\n".join(lines)

    def _current_cb_name(self) -> str:
        """Get the current callback name based on the current state."""
        return self._get_cb_name(self.state)

    def _get_cb_name(self, state: str) -> str:
        """Get the callback name for a given state."""
        return f"CB_{state}"
