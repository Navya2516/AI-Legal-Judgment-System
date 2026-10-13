import json
from collections import Counter

DATASET_PATH = "data/evaluation/IndicLegalQA Dataset_10K_Revised.json"


def main():
    # Load dataset
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    print("=" * 60)
    print("INDICLEGALQA DATASET INSPECTION")
    print("=" * 60)

    # Basic information
    print(f"Dataset type       : {type(data).__name__}")
    print(f"Number of records  : {len(data)}")

    # Check required fields
    required_fields = {
        "case_name",
        "judgement_date",
        "question",
        "answer"
    }

    field_counts = Counter()

    for record in data:
        field_counts.update(record.keys())

    print("\nFields found:")
    for field, count in field_counts.items():
        print(f"  {field}: {count}/{len(data)}")

    # Check missing values
    print("\nMissing values:")

    for field in required_fields:
        missing = sum(
            1 for record in data
            if not record.get(field)
        )

        print(f"  {field}: {missing}")

    # Unique cases
    unique_cases = set(
        record["case_name"]
        for record in data
        if record.get("case_name")
    )

    print(f"\nUnique cases       : {len(unique_cases)}")

    # Display sample
    print("\nSample record:")
    print("-" * 60)

    sample = data[0]

    for key, value in sample.items():
        print(f"{key}: {value}")

    print("=" * 60)
    print("Inspection completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()