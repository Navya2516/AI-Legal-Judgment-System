import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
model = os.getenv("GEMINI_MODEL")

print("=" * 60)
print("GEMINI DIAGNOSTIC")
print("=" * 60)

print("API KEY FOUND :", bool(api_key))

if api_key:
    print("API KEY PREFIX:", api_key[:8] + "...")

print("MODEL         :", model)

print("\nCreating Gemini client...")

client = genai.Client(
    api_key=api_key
)

print("Client created.")

# ------------------------------------------------------------
# TEST 1 — List models
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TEST 1: LIST MODELS")
print("=" * 60)

try:

    models = client.models.list()

    found = False

    for m in models:

        name = getattr(
            m,
            "name",
            ""
        )

        if model and model in name:

            print("FOUND MODEL:", name)

            supported = getattr(
                m,
                "supported_actions",
                None
            )

            print(
                "SUPPORTED ACTIONS:",
                supported
            )

            found = True

            break

    if not found:
        print(
            "WARNING: Requested model was not found "
            "in the returned model list."
        )

except Exception as e:

    print(
        "MODEL LIST ERROR:",
        type(e).__name__
    )

    print(
        str(e)
    )


# ------------------------------------------------------------
# TEST 2 — EXACT SAME MODEL USED BY SUMMARY SERVICE
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TEST 2: DIRECT GENERATE CONTENT")
print("=" * 60)

try:

    response = client.models.generate_content(
        model=model,
        contents="Say exactly: GEMINI SUMMARY TEST WORKED"
    )

    print("\nSUCCESS!")

    print(
        "MODEL RESPONSE:"
    )

    print(
        response.text
    )

except Exception as e:

    print("\nFAILED!")

    print(
        "ERROR TYPE:",
        type(e).__name__
    )

    print(
        "ERROR:"
    )

    print(
        str(e)
    )


print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)