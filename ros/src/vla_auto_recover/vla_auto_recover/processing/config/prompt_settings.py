from .system_settings import State, ADR, RDR, VDR

# ===========================================================================================
RUNNING_ACTION_ID = 0
RUNNING_LANGUAGE_INSTRUCTION = "move blocks from tray to matching dishes."

# ===========================================================================================
RECOVERY_ACTION_LIST = {
    1: {"class": "ANOMALY RECOVERY ACTION", "situation": "Misplaced blocks on different color dishes", "language_instruction": "Relocate any misplaced blocks to their matching dishes."},
    2: {"class": "ANOMALY RECOVERY ACTION", "situation": "Stacked dish on the other dish", "language_instruction": "Lift the stacked dish and set it down on the table."},
}  # fmt: skip

ACTION_LIST = {RUNNING_ACTION_ID: {"class": "NORMAL ACTION", "situation": "Moving blocks to matching dishes", "language_instruction": RUNNING_LANGUAGE_INSTRUCTION}} | RECOVERY_ACTION_LIST




# ===========================================================================================
GENERAL_PROMPT = f"""
あなたは、ロボットアームに対して指令を出す役割を担っています。
現在、<normal_action>{RUNNING_LANGUAGE_INSTRUCTION}<normal_action>を実行しています。
ロボットアームが実行しているタスクは、トレイからブロックを同じ色の皿に移動するタスクに関するものです。

# 正常状態とは
正常な状態とは、trayにblockがあること、blockが同じ色のdishに正しく配置されていることです。
blockがtrayにあることは異常なことではなく、初期状態です。

\n
"""

# ===========================================================================================
CB_RUNNING_PROMPT = (
    GENERAL_PROMPT
    + """
<recovery_action_list>
{recovery_action_list}
</recovery_action_list>

与えられた画像は、ロボットアームの現在の状態を示します。
action_id=0の`normal_action`を実行しているのですが、行動の実行に際して異常状態は画像の中に存在しますか？
また、action_id=0が完了している場合は、detection_result=`COMPLETION`だと報告してください。
action_id=0が問題なく実行できる場合は、detection_result=`NORMAL`, action_id=0を指定してください。
もし異常状態がある場合には、detection_result=`ANOMALY`として、解決する方法を下記の`recovery_action_list`から選択してください。
出力はJSON Schemaに従ってください。
"""
)

# ===========================================================================================
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

# ===========================================================================================
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
</recovery_action_list>

与えられた画像は、復帰行動後の状態を示します。
他にも異常状態は残っていないでしょうか？

【重要な制約】
- 異常状態が完全に解決されている場合: detection_result="SOLVED", action_id=0 を指定してください
- まだ異常状態が残っている場合: detection_result="UNSOLVED", action_id>=1以上を選択してください。

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
