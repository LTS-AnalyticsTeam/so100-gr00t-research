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
    # MODEL = "gpt-4o"

    USE_REFERENCE_IMAGES = False
    USE_OBJECT_DETECTION = True

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
        prompt = ps.CB_RUNNING_PROMPT.format(
            language_instruction=ps.RUNNING_LANGUAGE_INSTRUCTION,
            recovery_action_list=json.dumps(
                ps.RECOVERY_ACTION_LIST, indent=2, ensure_ascii=False
            ),
        )
        response = self._CB(
            prompt=prompt,
            observation_images=input_data.images,
            json_schema=ps.CB_RUNNING_JSON_SCHEMA,
        )
        output_data = self._parse_CB_RUNNING_response(response)
        return output_data

    def CB_RECOVERY(self, input_data: CB_InputIF) -> CB_OutputIF:
        """Check if the system is recovering from an anomaly"""
        # Get action instruction based on action_id
        language_instruction = ps.RECOVERY_ACTION_LIST[input_data.action_id]["language_instruction"]
        # messagesの作成
        prompt = ps.CB_RECOVERY_PROMPT.format(
            language_instruction=language_instruction,
        )
        response = self._CB(
            prompt=prompt,
            observation_images=input_data.images,
            json_schema=ps.CB_RECOVERY_JSON_SCHEMA,
        )
        output_data = self._parse_CB_RECOVERY_response(response)
        # CB_RECOVERYではアクションを変えない。
        output_data.action_id = input_data.action_id
        return output_data

    def CB_VERIFICATION(self, input_data: CB_InputIF) -> CB_OutputIF:
        """Verify if the anomaly has been resolved"""
        prompt = ps.CB_VERIFICATION_PROMPT.format(
            recovery_action_list=json.dumps(
                ps.RECOVERY_ACTION_LIST, indent=2, ensure_ascii=False
            ),
        )
        response = self._CB(
            prompt=prompt,
            observation_images=input_data.images,
            json_schema=ps.CB_VERIFICATION_JSON_SCHEMA,
        )
        output_data = self._parse_CB_VERIFICATION_response(response)
        return output_data

    def CB_END(self, input_data: CB_InputIF) -> CB_OutputIF:
        return CB_OutputIF()
    
    def _CB(self, prompt: str, observation_images: list[np.ndarray], json_schema: dict) -> None:
        """Placeholder for CB method"""

        openai_images = [
            ps.transform_np_image_to_openai(img) for img in observation_images
        ]        
        # messagesの作成
        content = []
        # promptの追加
        content.append({"type": "text", "text": prompt})

        # 観測画像の追加
        for i, img in enumerate(openai_images, start=1):
            content.append({"type": "text", "text": f"Observation Image No.{i}"})
            content.append({"type": "image_url", "image_url": {"url": img}})

        # 参照画像の追加
        if self.USE_REFERENCE_IMAGES:
            for i, img in enumerate(ps.NORMAL_IMAGES, start=1):
                content.append({"type": "text", "text": f"Normal Reference Image No.{i}"})
                content.append({"type": "image_url", "image_url": {"url": img}})
        
            for i, img in enumerate(ps.COMPLETION_IMAGES, start=1):
                content.append({"type": "text", "text": f"Completion Reference Image No.{i}"})
                content.append({"type": "image_url", "image_url": {"url": img}})
            
                    
            content.append({"type": "text", "text": "`Observation Images`と`Reference Images`の比較に基づいて、判断を行ってください。"})
        
        if self.USE_OBJECT_DETECTION:
            # 状態定義の追加
            content.append({"type": "text", "text": ps.DEFINITIONS_OF_STATE})
            
            # 物体検出を行う
            object_detection_result = self._object_detection(observation_images)
            content.append({"type": "text", "text": f"Object Detection Result: {object_detection_result}"})
            content.append({"type": "text", "text": "Please use the object detection result to make your decision."})

        print(f"content: \n{[c for c in content if c['type'] == 'text']}")
        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[{"role": "user", "content": content}],
            response_format={
                "type": "json_schema",
                "json_schema": json_schema,
            },
        )
        print(f"response: \n{response.choices[0].message.content}")
        return response

    def _object_detection(self, observation_images: list[np.ndarray]) -> str:
        """Detect objects in the observation images using VLM"""

        center_image = ps.transform_np_image_to_openai(observation_images[0])        
        content = [
            {"type": "text", "text": ps.OBJECT_DETECTION_PROMPT},
            {"type": "image_url", "image_url": {"url": center_image}},
            {"type": "text", "text": json.dumps(ps.OBJECT_DETECTION_SCHEMA)},
            {"type": "text", "text": "出力する前に状況について深く考察してから出力してください。"},
            {"type": "text", "text": "途中経過の思考も出力しつつ、最終的な出力はJSON Schemaに従ってください。"},
        ]
        print(f"content: \n{[c for c in content if c['type'] == 'text']}")
        response = self.client.chat.completions.create(
            model=self.MODEL,
            messages=[{"role": "user", "content": content}],
        )
        print(f"response: \n{response.choices[0].message.content}")
        return response.choices[0].message.content


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

