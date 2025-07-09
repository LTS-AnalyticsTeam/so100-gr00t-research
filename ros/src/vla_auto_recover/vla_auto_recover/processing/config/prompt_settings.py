from .system_settings import State, ADR, RDR, VDR
import cv2
import base64
import json
import numpy as np
from pathlib import Path

def transform_np_image_to_openai(image: np.ndarray) -> str:
    """Convert OpenCV image to base64 format for OpenAI API"""
    _, buffer = cv2.imencode(".jpg", image)
    base64_image = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_image}"

def transform_image_path_to_openai(image_path: Path) -> str:
    """Convert OpenCV image to base64 format for OpenAI API"""
    image = cv2.imread(str(image_path))
    return transform_np_image_to_openai(image)

# ===========================================================================================
ACTION_END_ID = -1
RUNNING_ACTION_ID = 0
RUNNING_LANGUAGE_INSTRUCTION = "move blocks from tray to matching dishes."

# ===========================================================================================
RECOVERY_ACTION_LIST = {
    1: {"class": "ANOMALY RECOVERY ACTION", "situation": "Stacked dish on the other dish", "language_instruction": "Unstack the dishes and arrange them individually on the table."},
    2: {"class": "ANOMALY RECOVERY ACTION", "situation": "Misplaced blocks on different color dishes", "language_instruction": "Relocate every red block to the red dish and every blue block to the blue dish, correcting any placement errors."},
}  # fmt: skip

ACTION_LIST = {RUNNING_ACTION_ID: {"class": "NORMAL ACTION", "situation": "Moving blocks to matching dishes", "language_instruction": RUNNING_LANGUAGE_INSTRUCTION}} | RECOVERY_ACTION_LIST
# ===========================================================================================
REFERENCE_IMAGE_DIR = Path("/workspace/ros/src/vla_auto_recover/vla_auto_recover/processing/config/reference_image")
IMAGE_START_CENTER_CAM = transform_image_path_to_openai(REFERENCE_IMAGE_DIR.joinpath("start", "center_cam.png"))
IMAGE_START_RIGHT_CAM = transform_image_path_to_openai(REFERENCE_IMAGE_DIR.joinpath("start", "right_cam.png"))
IMAGE_MIDDLE_CENTER_CAM = transform_image_path_to_openai(REFERENCE_IMAGE_DIR.joinpath("middle", "center_cam.png"))
IMAGE_MIDDLE_RIGHT_CAM = transform_image_path_to_openai(REFERENCE_IMAGE_DIR.joinpath("middle", "right_cam.png"))
IMAGE_END_CENTER_CAM = transform_image_path_to_openai(REFERENCE_IMAGE_DIR.joinpath("end", "center_cam.png"))
IMAGE_END_RIGHT_CAM = transform_image_path_to_openai(REFERENCE_IMAGE_DIR.joinpath("end", "right_cam.png"))


NORMAL_IMAGES = [
    IMAGE_START_CENTER_CAM,
    IMAGE_START_RIGHT_CAM,
    IMAGE_MIDDLE_CENTER_CAM,
    IMAGE_MIDDLE_RIGHT_CAM,
]

COMPLETION_IMAGES = [
    IMAGE_END_CENTER_CAM,
    IMAGE_END_RIGHT_CAM,
]


# ===========================================================================================
GENERAL_PROMPT = f"""
あなたは、ロボットアームに対して指令を出す役割を担っています。
現在、<normal_action>{RUNNING_LANGUAGE_INSTRUCTION}<normal_action>を実行しています。
ロボットアームが実行しているタスクは、トレイからブロックを同じ色の皿に移動するタスクに関するものです。
\n
"""

# ===========================================================================================
CB_RUNNING_PROMPT = (
    GENERAL_PROMPT
    + """
<recovery_action_list>
{recovery_action_list}
解除すべきタスクの優先度順に1,2,...と番号をつけています。複数のタスクがある場合は、優先度の高いものから順に実行してください。
</recovery_action_list>

与えられた画像は、ロボットアームの現在の状態を示します。
action_id=0の`normal_action`を実行しているのですが、行動の実行に際して異常状態は画像の中に存在しますか？
また、action_id=0が完了している場合は、detection_result=`COMPLETION`だと報告してください。
action_id=0が問題なく実行できる場合は、detection_result=`NORMAL`, action_id=0を指定してください。
もし異常状態がある場合には、detection_result=`ANOMALY`として、解決する方法を下記の`recovery_action_list`から選択してください。
出力はJSON Schemaに従ってください。
"""
)


CB_RUNNING_JSON_SCHEMA = {
    "name": "AnomalyDetectionResult",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "状態を判断した理由",
            },
            "action_id": {
                "type": "integer",
                "enum": [RUNNING_ACTION_ID] + list(RECOVERY_ACTION_LIST.keys()),
                "description": "次実行すべきaction_id",
            },
            "detection_result": {
                "type": "string",
                "enum": [ADR.NORMAL.value, ADR.ANOMALY.value, ADR.COMPLETION.value],
                "description": "action_id=0の実行に際しての状態",
            },
        },
        "required": ["reason", "action_id", "detection_result"],
        "additionalProperties": False,
    },
}

# ===========================================================================================
CB_RECOVERY_PROMPT = (
    GENERAL_PROMPT
    + """
<language_instruction>
{language_instruction}
</language_instruction>

与えられた画像は、ロボットアームの現在の状態を示します。
異常状態の復帰のために`language_instruction`を実行しているのですが、この行動はすでに完了していると思いますか？
完了済みの場合は、detection_result=`RECOVERED`を指定してください。
未完了の場合には、detection_result=`UNRECOVERED`を指定してください。
出力はJSON Schemaに従ってください。
"""
)


CB_RECOVERY_JSON_SCHEMA = {
    "name": "RecoveryStateResponse",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "状態を判断した理由",
            },
            "detection_result": {
                "type": "string",
                "enum": [RDR.RECOVERED.value, RDR.UNRECOVERED.value],
                "description": "language_instructionの完了状況",
            },
        },
        "required": ["reason", "detection_result"],
        "additionalProperties": False,
    },
}

# ===========================================================================================
CB_VERIFICATION_PROMPT = (
    GENERAL_PROMPT
    + """
<recovery_action_list>
{recovery_action_list}
解除すべきタスクの優先度順に1,2,...と番号をつけています。複数のタスクがある場合は、優先度の高いものから順に実行してください。
</recovery_action_list>

与えられた画像は、復帰行動後の状態を示します。
他にも異常状態は残っていないでしょうか？

【重要な制約】
- 異常状態が完全に解決されている場合: detection_result="SOLVED", action_id=0 を指定してください
- まだ異常状態が残っている場合: detection_result="UNSOLVED", action_id>=1以上を選択してください。

出力はJSON Schemaに従ってください。
"""
)

CB_VERIFICATION_JSON_SCHEMA = {
    "name": "VerificationStateResponse",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "状態を判断した理由",
            },
            "action_id": {
                "type": "integer",
                "enum": [RUNNING_ACTION_ID] + list(RECOVERY_ACTION_LIST.keys()),
                "description": "次実行すべきaction_id。異常状態がない場合は、0を指定してください。",
            },
            "detection_result": {
                "type": "string",
                "enum": [VDR.SOLVED.value, VDR.UNSOLVED.value],
                "description": "異常状態の解消状況",
            },
        },
        "required": ["reason", "action_id", "detection_result"],
        "additionalProperties": False,
    },
}

# ===========================================================================================

DEFINITIONS_OF_STATE= """
# 正常な状態とは
正常な状態とは、trayにblockがあること、blockが同じ色のdishに正しく配置されていることです。
blockがtrayにあることは異常なことではなく、初期状態です。

具体的には次のような状態は正常と言えます。
- 皿が分離して配置されている。
- 赤い皿の上には赤いブロックのみが載っている
- 青い皿の上には青いブロックのみが載っている
- 銀色のトレイは、ブロックが乗っていても乗っていなくても良い。

正常な場合のJSONの例を以下に示します。
```json
{
    "ARE_DISHES_SEPARATE": true,
    "ON_RED_DISH": {
        "RED_BLOCK": ">=0",
        "BLUE_BLOCK": "==0"
    },
    "ON_BLUE_DISH": {
        "RED_BLOCK": "==0",
        "BLUE_BLOCK": ">=0"
    },
    "ON_SILVER_TRAY": {
        "RED_BLOCK": ">=0",
        "BLUE_BLOCK": ">=0"
    }
}
```

# 異常の状態とは？
異常な状態とは、trayにblockがない、またはblockが異なる色のdishに不適切に配置されていることを指します。
具体的には次のような状態は異常と言えます。
- 皿が重なって配置されている。
- 赤い皿の上に青いブロックが載っている
- 青い皿の上に赤いブロックが載っている
- 銀色のトレイにブロックがない。
異常な場合のJSONの例を以下に示します。
```json
{
    "ARE_DISHES_SEPARATE": false,
    "ON_RED_DISH": {
        "RED_BLOCK": ">=0",
        "BLUE_BLOCK": ">=1"
    },
    "ON_BLUE_DISH": {
        "RED_BLOCK": ">=1",
        "BLUE_BLOCK": ">=0"
    },
    "ON_SILVER_TRAY": {
        "RED_BLOCK": ">=0",
        "BLUE_BLOCK": ">=0"
    }
}
```

# 完了の状態とは
銀色のトレイにある、すべてのブロックが、赤い皿または青い皿に移動されている状態を指します。
また、このとき皿とブロックの色が同じになるように配置されている必要があります。
完了の状態のJSONの例を以下に示します。
```json
{
    "ARE_DISHES_SEPARATE": true,
    "ON_RED_DISH": {
        "RED_BLOCK": ">=0",
        "BLUE_BLOCK": "==0"
    },
    "ON_BLUE_DISH": {
        "RED_BLOCK": "==0",
        "BLUE_BLOCK": ">=0"
    },
    "ON_SILVER_TRAY": {
        "RED_BLOCK": "==0",
        "BLUE_BLOCK": "==0"
    }
}
```
"""

# ===========================================================================================
OBJECT_DETECTION_PROMPT = """
画像内の状態をJSON Schemaに従ってJSON形式で出力してください。
赤の皿、青の皿、銀のトレーが存在しており、その上に、赤のブロックと青のブロックが載っています。

ON_RED_DISH, ON_BLUE_DISH, ON_SILVER_TRAYの各フィールドは、各皿の上にあるブロックの数を示します。
"""

OBJECT_DETECTION_SCHEMA = {
    "name": "DishBlockCountResponse",
    "schema": {
        "title": "Dish Block Count with Stack Flag",
        "type": "object",
        "properties": {
            "ARE_DISHES_SEPARATE": {
                "type": "boolean",
                "description": "青い皿と赤い皿が離れて配置されているか記述する。離れている場合はtrue。",
            },

            # ── RED DISH ────────────────────────────────
            "ON_RED_DISH": {
                "type": "object",
                "description": "赤い皿に載っているブロック数",
                "properties": {
                    "RED_BLOCK": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "赤ブロックの個数"
                    },
                    "BLUE_BLOCK": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "青ブロックの個数"
                    }
                },
                "required": ["RED_BLOCK", "BLUE_BLOCK"],
                "additionalProperties": False
            },

            # ── BLUE DISH ───────────────────────────────
            "ON_BLUE_DISH": {
                "type": "object",
                "description": "青い皿に載っているブロック数",
                "properties": {
                    "RED_BLOCK": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "赤ブロックの個数"
                    },
                    "BLUE_BLOCK": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "青ブロックの個数"
                    }
                },
                "required": ["RED_BLOCK", "BLUE_BLOCK"],
                "additionalProperties": False
            },

            # ── SILVER TRAY ─────────────────────────────
            "ON_SILVER_TRAY": {
                "type": "object",
                "description": "銀のトレイに載っているブロック数",
                "properties": {
                    "RED_BLOCK": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "赤ブロックの個数"
                    },
                    "BLUE_BLOCK": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "青ブロックの個数"
                    }
                },
                "required": ["RED_BLOCK", "BLUE_BLOCK"],
                "additionalProperties": False
            }
        },
        "required": [
            "ARE_DISHES_SEPARATE",
            "ON_RED_DISH",
            "ON_BLUE_DISH",
            "ON_SILVER_TRAY"
        ],
        "additionalProperties": False
    }
}
# ===========================================================================================