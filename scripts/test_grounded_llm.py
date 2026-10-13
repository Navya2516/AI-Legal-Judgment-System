import ollama

MODEL_NAME = "qwen2.5:1.5b"

question = """
What is the Basic Structure Doctrine according to the
Kesavananda Bharati judgment?
"""

legal_evidence = """
The Supreme Court considered the extent of Parliament's power
to amend the Constitution under Article 368. The case concerned
whether constitutional amendments could alter fundamental
features of the Constitution.

The judgment is associated with the Basic Structure Doctrine,
under which Parliament's power to amend the Constitution is
subject to limitations concerning the Constitution's basic
structure.

The Court considered the validity and scope of constitutional
amendments, including the 24th, 25th and 29th Amendments.
"""

prompt = f"""
You are an evidence-grounded legal judgment assistant.

IMPORTANT RULES:
1. Answer ONLY using the legal evidence provided below.
2. Do not use outside knowledge.
3. Do not invent legal facts.
4. If the evidence does not contain the answer, say:
   "The provided judgment evidence does not contain enough
   information to answer this question."
5. Do not claim that a provision, doctrine or fact exists
   unless supported by the evidence.
6. Explain the answer in simple language.
7. Do not provide legal advice.

LEGAL EVIDENCE:
{legal_evidence}

USER QUESTION:
{question}

Answer:
"""

response = ollama.chat(
    model=MODEL_NAME,
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print("\n===== GROUNDED LOCAL LLM RESPONSE =====\n")
print(response["message"]["content"])