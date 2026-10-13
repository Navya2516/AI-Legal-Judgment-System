import json
import os

folder = "data/evaluation"

for filename in os.listdir(folder):

    if not filename.endswith(".json"):
        continue

    path = os.path.join(folder, filename)

    print("\n" + "=" * 60)
    print(filename)

    if os.path.getsize(path) == 0:
        print("STATUS: EMPTY")
        continue

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        print("STATUS: VALID")
        print("TYPE:", type(data).__name__)

        if isinstance(data, dict):
            print("KEYS:", list(data.keys())[:20])

        elif isinstance(data, list):
            print("ITEMS:", len(data))

            if data:
                print("FIRST ITEM:")
                print(data[0])

    except Exception as e:
        print("STATUS: INVALID")
        print("ERROR:", e)