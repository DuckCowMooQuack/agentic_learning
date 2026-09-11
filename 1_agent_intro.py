import json
import re
import os
import litellm
from litellm import completion
from typing import List, Dict

model = "openai/gpt-4o"

def get_user_input():
    return input("Enter the function you want the agent to create: ")


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


def print_token_usage(total_tokens):
    max_context = litellm.get_model_info(model)["max_input_tokens"]
    remaining = max_context - total_tokens
    title_box = make_border([
        "Token Usage"
    ])
    lines = title_box + [
        f"Context limit: {max_context}",
        f"  Tokens used: {total_tokens}",
        f"  Tokens left: {remaining}",
    ]

    print_with_border(lines)


def print_response_message(response_message):
    title_box = make_border([
        "Response Message"
    ])
    lines = title_box + response_message

    print_with_border(lines)


def gen_resp(messages: List[Dict]) -> str:
    """Call LLM to get response"""
    response = completion(
        model=model,
        messages=messages,
        max_tokens=1024
    )

    return response.choices[0].message.content, response.usage.total_tokens

input = get_user_input()

# Initial message asking for a function
messages = [
    {
        "role": "system",
        "content": "You are an expert software engineer that prefers "
                   "functional programming. When given a prompt, "
                   "expect a function name/desciprtion from the user." 
                   "Write the requested python function. Return the "
                   "python function inside ```<code>``` block, where "
                   "<code> is the function you're writing for the user. "
                   "Include only one code block in your response."
    },
    {
        "role": "user",
        "content": f"{input}"
    }
]

response_message, total_tokens = gen_resp(messages)

block_search = re.search(r"```(?:\w+)?\n(.*?)```", response_message, re.DOTALL)

response_code_block = []

if block_search:
    response_code_block = block_search.group(1).splitlines()

print_response_message(response_message.splitlines())

messages.extend([
    {
        "role": "assistant",
        "content": "\n".join(response_code_block)
    },
    {
        "role": "user",
        "content": "Add documentation to the previoussly generated function "
                   "including Function description, Parameter descriptions, Return value "
                   "description, Example usage, and Edge cases.Return the "
                   "python function that includes the documentation inside "
                   "```<code>``` block, where <code> is the function you're "
                   "writing for the user. Include only one code block in your response."
    }
])

response_message, total_tokens = gen_resp(messages)

block_search = re.search(r"```(?:\w+)?\n(.*?)```", response_message, re.DOTALL)

if block_search:
    response_code_block = block_search.group(1).splitlines()

print_response_message(response_message.splitlines())

messages.extend([
    {
        "role": "assistant",
        "content": "\n".join(response_code_block)
    },
    {
    "role": "user",
    "content": "Add unit test cases using Python's unittest framework "
               "to the previoussly generated function that should include basic "
               "functionality, edge cases, and error cases. "
               "Return the python function that includes the unittests inside "
               "```<code>``` block, where <code> is the function you're "
               "writing for the user. Include only one code block in your response."
    }
])

response_message, total_tokens = gen_resp(messages)

block_search = re.search(r"```(?:\w+)?\n(.*?)```", response_message, re.DOTALL)

if block_search:
    response_code_block = block_search.group(1).splitlines()

print_response_message(response_message.splitlines())
print_token_usage(total_tokens)
