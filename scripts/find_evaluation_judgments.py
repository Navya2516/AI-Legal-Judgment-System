import json
import os
import requests
import pandas as pd

MANIFEST = "data/evaluation/evaluation_cases_manifest.json"

BASE_URL = (
    "https://indian-supreme-court-judgments.s3.amazonaws.com/"
    "metadata/parquet/year={}/metadata.parquet"
)

with open(MANIFEST, "r", encoding="utf-8") as f:
    cases = json.load(f)

case = cases[0]

year = int(case["judgement_date"][-4:])

print("Evaluation case:")
print("Case:", case["case_name"])
print("Date:", case["judgement_date"])
print("Year:", year)

url = BASE_URL.format(year)

print("\nDownloading metadata:")
print(url)

response = requests.get(url, timeout=120)

print("HTTP status:", response.status_code)

if not response.ok:
    print(response.text[:500])
    raise SystemExit(1)

metadata_file = f"data/evaluation/metadata_{year}.parquet"

with open(metadata_file, "wb") as f:
    f.write(response.content)

print("\nSaved:", metadata_file)
print("Size:", round(len(response.content) / 1024 / 1024, 2), "MB")

df = pd.read_parquet(metadata_file)

print("\nColumns:")
print(list(df.columns))

print("\nNumber of records:", len(df))

print("\nFirst 5 records:")
print(df.head().to_string())