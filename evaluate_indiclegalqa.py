import json
import os
import random
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "data/evaluation/IndicLegalQA Dataset_10K_Revised.json"

OUTPUT_PATH = (
    "data/evaluation/indiclegalqa_evaluation_results.json"
)

SAMPLE_SIZE = 100
RANDOM_SEED = 42


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}"
        )

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            "IndicLegalQA dataset must contain a JSON list."
        )

    if not data:
        raise ValueError(
            "IndicLegalQA dataset is empty."
        )

    return data


# ============================================================
# SELECT EVALUATION QUESTIONS
# ============================================================

def select_questions(
    dataset,
    sample_size=SAMPLE_SIZE
):

    random.seed(RANDOM_SEED)

    # --------------------------------------------------------
    # Use all questions if dataset is smaller
    # --------------------------------------------------------

    if len(dataset) <= sample_size:
        return dataset

    # --------------------------------------------------------
    # Try to obtain questions from different cases
    # --------------------------------------------------------

    cases = {}

    for item in dataset:

        case_name = item.get(
            "case_name",
            "Unknown Case"
        )

        cases.setdefault(
            case_name,
            []
        ).append(item)

    case_names = list(cases.keys())

    random.shuffle(case_names)

    selected = []

    # --------------------------------------------------------
    # First pass:
    # one question per case
    # --------------------------------------------------------

    for case_name in case_names:

        if len(selected) >= sample_size:
            break

        selected.append(
            random.choice(
                cases[case_name]
            )
        )

    # --------------------------------------------------------
    # Second pass if necessary
    # --------------------------------------------------------

    remaining = [
        item
        for item in dataset
        if item not in selected
    ]

    random.shuffle(remaining)

    while (
        len(selected) < sample_size
        and remaining
    ):

        selected.append(
            remaining.pop()
        )

    return selected


# ============================================================
# PRINT EVALUATION SET
# ============================================================

def print_evaluation_set(
    questions
):

    print()
    print("=" * 60)
    print("INDICLEGALQA EVALUATION SET")
    print("=" * 60)

    print(
        f"Questions selected : {len(questions)}"
    )

    print(
        f"Random seed        : {RANDOM_SEED}"
    )

    print()

    for i, item in enumerate(
        questions[:10],
        start=1
    ):

        print(
            f"{i}. {item.get('question', '')}"
        )

        print(
            f"   Case: {item.get('case_name', '')}"
        )

        print()


# ============================================================
# SAVE EVALUATION SET
# ============================================================

def save_evaluation_set(
    questions
):

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    result = {

        "evaluation_name":
            "IndicLegalQA Baseline Evaluation",

        "dataset":
            "IndicLegalQA Dataset_10K_Revised.json",

        "total_dataset_questions":
            None,

        "evaluation_questions":
            len(questions),

        "random_seed":
            RANDOM_SEED,

        "created_at":
            datetime.now().isoformat(),

        "status":
            "evaluation_set_created",

        "questions":
            questions
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"Evaluation set saved to:\n{OUTPUT_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("AI LEGAL JUDGMENT SYSTEM")
    print("IndicLegalQA Evaluation")
    print("=" * 60)

    print()
    print("Loading dataset...")

    dataset = load_dataset()

    print(
        f"Dataset loaded successfully."
    )

    print(
        f"Total questions: {len(dataset)}"
    )

    print()

    print(
        f"Selecting {SAMPLE_SIZE} evaluation questions..."
    )

    questions = select_questions(
        dataset,
        SAMPLE_SIZE
    )

    print_evaluation_set(
        questions
    )

    save_evaluation_set(
        questions
    )

    print()
    print("=" * 60)
    print("STEP 1 COMPLETE")
    print("=" * 60)

    print()
    print(
        "Next step: connect these questions to your"
    )
    print(
        "existing RAG question-answering service."
    )


if __name__ == "__main__":
    main()