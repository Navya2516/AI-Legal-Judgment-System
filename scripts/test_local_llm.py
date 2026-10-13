import ollama

MODEL_NAME = "qwen2.5:1.5b"

question = """
Explain the Basic Structure Doctrine of the Indian Constitution
using simple language.
"""

response = ollama.chat(
    model=MODEL_NAME,
    messages=[
        {
            "role": "user",
            "content": question
        }
    ]
)

print("\n===== LOCAL LLM RESPONSE =====\n")
print(response["message"]["content"])