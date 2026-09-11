import argparse
import json
import re
import os
import litellm
from litellm import completion
from typing import List, Dict
from pathlib import Path

model = "openai/gpt-4o"

tools = [
    {
        "type": "function",
        "function": 
        {
            "name": "list_files",
            "description": "Returns a list of files in the directory.",
            "parameters": 
            {
                "type": "object",
                "properties": 
                {
                    "directory":
                    {
                        "type": "string",
                        "description": "Directory to list. Defaulits to current directory."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function":
        {
            "name": "read_file",
            "description": "Reads the content of a specified file in the directory.",
            "parameters": 
            {
                "type": "object",
                "properties": 
                {
                    "file_name": 
                    {
                        "type": "string"
                    },
                    "directory":
                    {
                        "type": "string",
                        "description": "Directory containing the file. Defaults to the current directory."
                    }
                },
                "required": ["file_name"]
            }
        }
    }
]

agent_rules = [{
    "role": "system",
    "content": 
        """
              You are an AI agent that can perform tasks by using available tools.

              If a user asks about files, list them before reading.
        """
}]


def list_files(directory="."):
    return [file.name for file in Path(directory).iterdir()]


def read_file(file, directory="."):
    with open(f"{directory}/{file}", "r") as file:
        contents = file.read()

    return(contents)


def execute_action(tool_name, args, total_tokens):
    if tool_name == "list_files":
        return list_files(
            args.get("directory", ".")
        )
    elif tool_name == "read_file":
        return read_file(
            args["file_name"],
            args.get("directory", ".")
        )
    else:
        return f"Unknown tool: {tool_name}"


def get_user_input():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "user_input",
        nargs="?",
        default="Read file[1-4] in /tmp",
        help="Task for agent"
    )

    args = parser.parse_args()

    if args.user_input:
        return args.user_input

    return input("What would you like to do? ")


def result_to_string(action_result):
    if isinstance(action_result, list):
        return "\n".join(str(result) for result in action_result)

    return str(action_result)


def get_bordered_message(title, message):
    if isinstance(message, str):
        message = message.splitlines()

    return make_border(
        message,
        title
    )


def make_border(lines, title=None):
    width = max(len(line) for line in lines)

    if title:
        width = max(width, len(title) + 2)

        top_border = (
            "┌─ "
            + title
            + " "
            + "─" * (width - len(title) - 1)
            + "┐"
        )
    else:
        top_border = (
            "┌"
            + "─" * (width + 2)
            + "┐"
        )

    bordered = [
        top_border,
        f"│ {'':<{width}} │"
    ]

    for line in lines:
        bordered.append(
            f"│ {line:<{width}} │"
        )

    bordered.append(
        "└"
        + "─" * (width + 2)
        + "┘"
    )

    return bordered


def print_with_border(lines):
    for line in make_border(lines):
        print(line)


def print_bordered_message(title, message):
    for line in make_border(
        message,
        title
        ):
            print(line)


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
        tools=tools,
        max_tokens=1024
    )

    return response.choices[0].message, response.usage.total_tokens


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

    iteration_output = []

    # 1. Generate response

    response_message, total_tokens = gen_resp(messages)

    # 2. Check for tool calls

    tool_calls = response_message.tool_calls

    if not tool_calls:
        iteration_output.extend(
            get_bordered_message(
                "Response Message:",
                response_message.content
            )
        )
        
        iteration_output.append("")

        token_summary = get_token_usage_summary(total_tokens)

        print_bordered_message(
            f"Loop Iteration {iteration}:",
            iteration_output
        )

        print_bordered_message(
            f"Token Usage",
            token_summary
        )

        break

    tool_call = tool_calls[0]

    action_tool_name = tool_call.function.name
    action_args = json.loads(tool_call.function.arguments)
    action_args_json = json.dumps(
        action_args,
        indent=4
    )
    action_json = json.dumps(
        tool_call.model_dump(),
        indent=4
    )

    action_summary = [
        "Action JSON: "
    ]

    action_summary.extend(action_json.splitlines())

    action_summary.append("Action Args:")

    action_summary.extend(action_args_json.splitlines())

    iteration_output.extend(
        get_bordered_message(
            "Action Summary:",
            action_summary
        )
    )

    iteration_output.append("")

    # 3. Execute action

    action_result = execute_action(
        action_tool_name,
        action_args,
        total_tokens
    )

    # 4. Convert action_result to string

    action_result_string = result_to_string(action_result)

    iteration_output.extend(
        get_bordered_message(
            "Action Summary:",
            action_summary
        )
    )

    print_bordered_message(
        f"Loop Iteration {iteration}:",
        iteration_output
    )

    # 5. Extend message with response_message.content and action_result_string

    messages.extend([
        {
            "role": "assistant",
            "content": response_message.content,
            "tool_calls": [
                tool_call.model_dump()
            ]
        },
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": action_result_string
        }
    ])

    iteration += 1
