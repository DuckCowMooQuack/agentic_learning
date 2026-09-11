import json
import re
import os
import litellm
from litellm import completion
from typing import List, Dict
from pathlib import Path

model = "openai/gpt-4o"
agent_rules = [{
    "role": "system",
    "content": """
              You are an AI agent that can perform tasks by using available tools.

              Available tools:
              - list_files() -> List[str]: List all files in the current directory.
              - read_file(file_name: str) -> str: Read the content of a file
              - terminate(message: str): End the agen loop and print a summary to the user.
              
              If a user asks about files, list them before reading.

              Every response MUST have an action.
              Respond in this format. Make sure to include "action = {":
              ```
              action = {
                  "tool_name": "insert tool name",
                  "args": {...fill in any required arguments here...}
              }
              """
}]


def list_files():
    return [file.name for file in Path(".").iterdir()]


def read_file(file):
    with open(file, "r") as file:
        contents = file.read()

    return(contents)


def terminate(exit_message, total_tokens):
    token_summary = get_token_usage_summary(total_tokens)
    print_bordered_message("Token Usage:", token_summary)
    print_bordered_message("Exit Message:", exit_message)
    exit()


def execute_action(tool_name, args, total_tokens):
    if tool_name == "list_files":
        return list_files()
    elif tool_name == "read_file":
        return read_file(args["file_name"])
    elif tool_name == "terminate":
        terminate(args["message"], total_tokens)
    else:
        return f"Unknown tool: {tool_name}"


def get_user_input():
    return input("What would you like to do? ")


def result_to_string(action_result):
    if isinstance(action_result, list):
        return "\n".join(str(result) for result in action_result)

    return str(action_result)

def parse_action(response_message):
    match = re.search(r"\baction\s*=\s*(\{)", response_message)

    if not match:
        return None, None, None

    json_start = match.start(1)
    decoder = json.JSONDecoder()

    try:
        action, _ = decoder.raw_decode(response_message[json_start:])
    except json.JSONDecodeError:
        return None, None, None

    return (
        action,
        action.get("tool_name"),
        action.get("args")
    )

def make_border(lines):
    width = max(len(line) for line in lines)
    bordered = [
        "┌" + "─" * (width + 2) + "┐"
    ]

    for line in lines:
        bordered.append(f"│ {line:<{width}} │")

    bordered.append(
        "└" + "─" * (width + 2) + "┘"
    )

    return bordered


def print_with_border(lines):
    for line in make_border(lines):
        print(line)
    print()


def print_bordered_message(title, message):
    if isinstance(message, str):
        message = message.splitlines()

    title_box = make_border([
        title
    ])
    lines = title_box + message
    print_with_border(lines)


def get_token_usage_summary(total_tokens):
    max_context = litellm.get_model_info(model)["max_input_tokens"]
    remaining = max_context - total_tokens
    token_summary = [
        f"Context limit: {max_context}",
        f"  Tokens used: {total_tokens}",
        f"  Tokens left: {remaining}",
    ]

    return token_summary


def gen_resp(messages: List[Dict]) -> str:
    """Call LLM to get response"""
    response = completion(
        model=model,
        messages=messages,
        max_tokens=1024
    )

    return response.choices[0].message.content, response.usage.total_tokens


# 1. Construct initial prompt

input = get_user_input()

# Initial message asking for a function
messages = agent_rules
messages.extend([
    {
        "role": "user",
        "content": f"{input}"
    }
])

iteration = 0
while True:

    # 2. Generate response

    response_message, total_tokens = gen_resp(messages)
    print_bordered_message("Response Message:", response_message)

    # 3. Parse Response

    # Evaluate response_message and find the action json
    action_json, action_tool_name, action_args = parse_action(response_message)

    action_summary = [
        f"Action JSON: {action_json}",
        f"Action Tool Name: {action_tool_name}",
        f"Action Args: {action_args}",
    ]

    print_bordered_message("Action Summary:", action_summary)

    # 4. Execute action

    action_result = execute_action(action_tool_name, action_args, total_tokens)

    # 5. Convert action_result to string

    action_result_string = result_to_string(action_result)
    print_bordered_message("Action Result:", action_result_string)

    messages.extend([
        {
            "role": "assistant",
            "content": response_message
        },
        {
            "role": "user",
            "content": f"Tool result:\n{action_result_string}"
            }
    ])

    iteration += 1
