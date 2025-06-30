from .system_state import State, ADR, RDR, VDR

# ===========================================================================================
GENERAL_PROMPT = """
あなたは、ロボットアームに対して指令を出す役割を担っています。
ロボットアームが実行しているタスクは、トレイからブロックを同じ色の皿に移動するタスクに関するものです。
以降の判断について検討してください。 \n
"""

# ===========================================================================================
NORMAL_LANGUAGE_INSTRUCTION = "move blocks from tray to matching dishes."

# ===========================================================================================
ACTION_LIST = {
    0: {"class": "NORMAL ACTION", "situation": "No anomaly", "language_instruction": NORMAL_LANGUAGE_INSTRUCTION},
    1: {"class": "ANOMALY RECOVERY ACTION", "situation": "Misplaced blocks on different color dishes", "language_instruction": "Relocate any misplaced blocks to their matching dishes."},
    2: {"class": "ANOMALY RECOVERY ACTION", "situation": "Stacked dish on the other dish", "language_instruction": "Lift the stacked dish and set it down on the table."},
}  # fmt: skip

# ===========================================================================================
CB_NORMAL_PROMPT = (
    GENERAL_PROMPT
    + f"\n<language_instruction>\n{NORMAL_LANGUAGE_INSTRUCTION}\n</language_instruction>\n"
    + """
<action_list>
{action_list}
</action_list>

与えられた画像は、ロボットアームの現在の状態を示します。
action_id=0の`language_instruction`を実行しているのですが、行動の実行に際して問題となる事象は画像の中に存在しますか？
また、action_id=0が完了している場合は、detection_result=`FINISHED`だと報告してください。
action_id=0が問題なく実行できる場合は、detection_result=`NORMAL`, action_id=0を指定してください。
もし問題がある場合には、detection_result=`ANOMALY`として、解決する方法を下記の`action_list`から選択してください。
出力はJSON Schemaに従ってください。
"""
)

# ===========================================================================================
CB_NORMAL_JSON_SCHEMA = {
    "name": "AnomalyDetectionResult",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "detection_result": {
                "type": "string",
                "enum": [ADR.NORMAL.value, ADR.ANOMALY.value, ADR.FINISHED.value],
                "description": "action_id=0の実行に際しての状態",
            },
            "action_id": {
                "type": "integer",
                "enum": list(ACTION_LIST.keys()),
                "description": "次実行すべきaction_id",
            },
            "reason": {
                "type": "string",
                "description": "状態を判断した理由",
            },
        },
        "required": ["detection_result", "action_id", "reason"],
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

# ===========================================================================================
CB_RECOVERY_JSON_SCHEMA = {
    "name": "RecoveryStateResponse",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "detection_result": {
                "type": "string",
                "enum": [RDR.RECOVERED.value, RDR.UNRECOVERED.value],
                "description": "language_instructionの完了状況",
            },
            "reason": {
                "type": "string",
                "description": "状態を判断した理由",
            },
        },
        "required": ["detection_result", "reason"],
        "additionalProperties": False,
    },
}

# ===========================================================================================
CB_VERIFICATION_PROMPT = (
    GENERAL_PROMPT
    + """
<language_instruction>
{language_instruction}
</language_instruction>

<action_list>
{action_list}
</action_list>

与えられた画像は、ロボットアームの現在の状態を示します。
復帰行動として`language_instruction`を実行した後の状態ですが、行動により問題は完全に解決されていると思いますか？

【重要な制約】
- 問題が完全に解決されている場合: detection_result="SOLVED", action_id=0 を指定してください
- まだ問題が残っている場合: detection_result="UNSOLVED", action_id>=1以上を選択してください。

出力はJSON Schemaに従ってください。
"""
)

# ===========================================================================================
CB_VERIFICATION_JSON_SCHEMA = {
    "name": "VerificationStateResponse",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "detection_result": {
                "type": "string",
                "enum": [VDR.SOLVED.value, VDR.UNSOLVED.value],
                "description": "language_instructionの完了状況",
            },
            "action_id": {
                "type": "integer",
                "enum": list(ACTION_LIST.keys()),
                "description": "次実行すべきaction_id",
            },
            "reason": {
                "type": "string",
                "description": "状態を判断した理由",
            },
        },
        "required": ["detection_result", "action_id", "reason"],
        "additionalProperties": False,
    },
}
