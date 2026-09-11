import litellm

response = completion(
    model="openai/gpt-4o",
    messages=messages,
    max_tokens=1024
)

print(f"Input tokens:  {response.usage.prompt_tokens}")
print(f"Output tokens: {response.usage.completion_tokens}")
print(f"Total tokens:  {response.usage.total_tokens}")

print(f"{response.choices[0].message.content}")
