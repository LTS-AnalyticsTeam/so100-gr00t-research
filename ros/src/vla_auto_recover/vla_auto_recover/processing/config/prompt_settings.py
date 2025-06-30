# ===========================================================================================
ACTION_LIST = [
    {"action_id": 0, "class": "NORMAL ACTION", "language_instruction": "move blocks from tray to matching dishes."},
    {"action_id": 1, "class": "ANOMALY RECOVERY ACTION", "language_instruction": "Relocate any misplaced blocks to their matching dishes."},
    {"action_id": 2, "class": "ANOMALY RECOVERY ACTION", "language_instruction": "Lift the stacked dish and set it down on the table."},
] # fmt: skip

# ===========================================================================================
ANOMALY_DETECTION_PROMPT = """
<action_instruction>
{action_instruction}
</action_instruction>

<action_list>
{action_list}
</action_list>

与えられた画像は、ロボットアームの現在の状態を示します。
`action_instruction`を実行しているのですが、行動の実行に際して問題となる事象は画像の中に存在しますか？
もし問題がある場合には、解決する方法を下記の`action_list`から選択してください。
出力はJSON Schemaに従ってください。
"""

# ===========================================================================================
RECOVERY_STATE_PROMPT = """
<action_instruction>
{action_instruction}
</action_instruction>

与えられた画像は、ロボットアームの現在の状態を示します。
異常状態の回復のために`action_instruction`を実行しているのですが、この行動はすでに完了していると思いますか？
もし問題がある場合には、解決する方法を下記の`action_list`から選択してください。
出力はJSON Schemaに従ってください。
"""

# ===========================================================================================
JSON_SCHEMA_ANOMALY_DETECTION = {
    "name": "AnomalyDetectionResponse",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "state": {"type": "string", "enum": ["NORMAL", "ANOMALY"]},
            "action_id": {"type": "integer"},
            "reason": {
                "type": "string",
                "description": "状態を判断した理由",
            },
        },
        "required": ["state", "action_id", "reason"],
        "additionalProperties": False,
    },
}


# ===========================================================================================
JSON_SCHEMA_RECOVERY_STATE = {
    "name": "RecoveryStateResponse",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "state": {"type": "string", "enum": ["NORMAL", "ANOMALY"]},
            "reason": {
                "type": "string",
                "description": "状態を判断した理由",
            },
        },
        "required": ["state", "reason"],
        "additionalProperties": False,
    },
}


if __name__ == "__main__":
    # output settings as text file
    with open("settings_output.txt", "w") as f:
        f.write("=" * 50 + "\n")
        f.write("         VLA AUTO RECOVER SETTINGS\n")
        f.write("=" * 50 + "\n\n")

        f.write("State Enum:\n")
        f.write("-" * 30 + "\n")
        for state in State:
            f.write(f"  • {state.value}\n")
        f.write("\n\n")

        f.write("Action List:\n")
        f.write("-" * 30 + "\n")
        for id, action in ACTION_LIST.items():
            f.write(f"  {id}: {action}\n")
        f.write("\n\n")

        f.write("Anomaly Detection Prompt:\n")
        f.write("-" * 30 + "\n")
        f.write(ANOMALY_DETECTION_PROMPT)
        f.write("\n\n")

        f.write("Recovery State Prompt:\n")
        f.write("-" * 30 + "\n")
        f.write(RECOVERY_STATE_PROMPT)
        f.write("\n\n")

        f.write("JSON Schema for Anomaly Detection:\n")
        f.write("-" * 30 + "\n")
        f.write(JSON_SCHEMA_ANOMALY_DETECTION)
        f.write("\n\n")

        f.write("JSON Schema for Recovery State:\n")
        f.write("-" * 30 + "\n")
        f.write(JSON_SCHEMA_RECOVERY_STATE)
        f.write("\n\n")

        f.write("=" * 50 + "\n")
        f.write("              END OF SETTINGS\n")
        f.write("=" * 50 + "\n")
