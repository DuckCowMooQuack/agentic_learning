#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import litellm
from litellm import completion


class AgentApp:
    def __init__(self):
        self.config = AgentConfig()
        self.environment = Environment()
        self.action_factory = ActionFactory(self.environment)
        self.registry = ActionRegistry()
        self.printer = Printer(self.config)
        self.input_reader = InputReader()

        self.registry.register_all(
            self.action_factory.create_all_actions()
        )

        self.agent = Agent(
            self.config,
            self.registry,
            self.printer
        )

    def run(self):
        user_input = self.input_reader.get_user_input()
        self.agent.run(user_input)


class AgentConfig:
    def __init__(self):
        self.model = "openai/gpt-4o"
        self.agent_rules = [{
            "role": "system",
            "content": 
                """
                You are an AI agent that can perform tasks by using available tools.
                """
        }]


class Environment:
    """ The things the program can actually do """
    def list_files(self, directory="."):
        return [
            file.name
            for file in Path(directory).iterdir()
        ]

    def read_file(self, file_name, directory="."):
        file_path = Path(directory) / file_name

        with open(file_path, "r") as file:
            return file.read()
    
    def count_characters(self, text):
        return len(text)

    def get_script_name(self):
        return Path(__file__).name


class Action:
    """ description + executable reference for one capability """
    def __init__(self, name, description, parameters, function):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.function = function

    def execute(self, args):
        return self.function(**args)

    def get_tool_definition(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }


class ActionRegistry:
    """ collection/dispatcher for all Actions """
    def __init__(self):
        self.actions = {}

    def register(self, action):
        self.actions[action.name] = action

    def register_all(self, actions):
        for action in actions:
            self.register(action)

    def execute(self, name, args):
        action = self.actions.get(name)

        if not action:
            return f"Unknown action: {name}"
        try:
            return action.execute(args)
        except Exception as error:
            return f"Action failed: {type(error).__name__}: {error}"

    def get_tools(self):
        return [
            action.get_tool_definition()
            for action in self.actions.values()
        ]


class ActionFactory:
    def __init__(self, environment):
        self.environment = environment

    def create_list_files_action(self):
        return Action(
            name="list_files",
            description="Returns a list of files in the directory.",
            parameters={
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Directory to list."
                    }
                },
                "required": []
            },
            function=self.environment.list_files
        )

    def create_read_file_action(self):
        return Action(
            name="read_file",
            description="Reads a file",
            parameters={
                "type": "object",
                "properties": {
                    "file_name": {
                        "type": "string",
                        "description": "Name of file to be read."
                    },
                    "directory": {
                        "type": "string",
                        "description": "Directory containing the file."
                    }
                },
                "required": ["file_name"]
            },
            function=self.environment.read_file
        )

    def create_count_characters_action(self):
        return Action(
            name="count_characters",
            description="Counts the number of characters in a string.",
            parameters={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to count characters in."
                    }
                },
                "required": ["text"]
            },
            function=self.environment.count_characters
        ) 

    def create_get_script_name_action(self):
        return Action(
            name="get_script_name",
            description="Returns the name of the script currently being run.",
            parameters={
                "type": "object",
                "properties": {},
                "required": []
            },
            function=self.environment.get_script_name
        )

    def create_all_actions(self):
        return [
            self.create_list_files_action(),
            self.create_read_file_action(),
            self.create_count_characters_action(),
            self.create_get_script_name_action()
        ]

class Agent:
    """Owns the conversation state and model calls."""
    def __init__(self, config, registry, printer):
        self.config = config
        self.messages = config.agent_rules.copy()
        self.registry = registry
        self.printer = printer
        self.iteration = 0

    def gen_resp(self):
        response = completion(
            model=self.config.model,
            messages=self.messages,
            tools=self.registry.get_tools(),
            max_tokens=1024
        )

        return response.choices[0].message, response.usage.total_tokens

    def handle_tool_call(self, response_message, tool_call):
        action_tool_name = tool_call.function.name
        action_args = json.loads(tool_call.function.arguments)

        action_result = self.registry.execute(
            action_tool_name,
            action_args
        )

        action_result_string = self.printer.result_to_string(action_result)
        
        self.messages.extend([
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

        return action_result_string

    def get_tool_call_summary(self, tool_call):
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

        return action_summary


    def print_final_response(self, response_message, total_tokens, iteration_output):
        token_summary = self.printer.get_token_usage_summary(total_tokens)
        iteration_output.extend(
            self.printer.get_bordered_message(
                "Response Message:",
                response_message.content
            )
        )
        iteration_output.append("")
        self.printer.print_bordered_message(
            f"Loop Iteration {self.iteration}:",
            iteration_output
        )
        self.printer.print_bordered_message(
            "Token Usage:",
            token_summary
        )


    def run_tool_iteration(self, response_message, tool_call, iteration_output):
        iteration_output.extend(
            self.printer.get_bordered_message(
                "Action Summary:",
                self.get_tool_call_summary(tool_call)
            )
        )
        iteration_output.append("")

        action_result_string = self.handle_tool_call(response_message, tool_call)

        iteration_output.extend(
            self.printer.get_bordered_message(
                "Action Result:",
                action_result_string
            )
        )

        self.printer.print_bordered_message(
             f"Loop Iteration {self.iteration}:",
             iteration_output
         )

    def run(self, user_input):
        self.messages.append({
            "role": "user",
            "content": user_input
        })

        while True:
            iteration_output = []
            response_message, total_tokens = self.gen_resp()
            tool_calls = response_message.tool_calls

            if not tool_calls:
                self.print_final_response(response_message, total_tokens, iteration_output)
                break

            self.run_tool_iteration(response_message, tool_calls[0], iteration_output)

            self.iteration += 1


class Printer:
    def __init__(self, config):
        self.config = config
    def get_bordered_message(self, title, message):
        if isinstance(message, str):
            message = message.splitlines()

        return self.make_border(
            message,
            title
        )


    def make_border(self, lines, title=None):
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


    def print_bordered_message(self, title, message):
        for line in self.get_bordered_message(title, message):
            print(line)

    def get_token_usage_summary(self, total_tokens):
        max_context = litellm.get_model_info(self.config.model)["max_input_tokens"]
        remaining = max_context - total_tokens
        token_summary = [
            f"Context limit: {max_context}",
            f"  Tokens used: {total_tokens}",
            f"  Tokens left: {remaining}",
        ]

        return token_summary


    def result_to_string(self, action_result):
        if isinstance(action_result, list):
            return "\n".join(str(result) for result in action_result)

        return str(action_result)


class InputReader:
    def get_user_input(self):
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

def main():
    app = AgentApp()
    app.run()

if __name__ == "__main__":
    main()
