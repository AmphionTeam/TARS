import argparse
import json

from xfinder.eval import Evaluator
from tqdm import tqdm
from huggingface_hub import snapshot_download

evaluator = Evaluator(
    model_name="xFinder-qwen1505",  # Model name
    inference_mode="local",  # Inference mode, 'local' or 'api'
    model_path_or_url=snapshot_download(
        "IAAR-Shanghai/xFinder-qwen1505"
    ),  # Anonymized model path or URL
)


def xfinder_check(question, options, answer, llm_output):
    # Define input for a single evaluation
    standard_answer_range = []
    for a, o in zip("ABCD", options):
        standard_answer_range.append([a, str(o)])
    standard_answer_range = json.dumps(standard_answer_range)

    # Perform single example evaluation
    result = evaluator.evaluate_single_item(
        question, llm_output, standard_answer_range, "alphabet_option", answer
    )
    return result[-1]


def main():
    parser = argparse.ArgumentParser(description="xFinder evaluation script")
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input JSON file"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output JSON file"
    )
    args = parser.parse_args()

    # Read input data
    with open(args.input, "r") as f:
        data = json.load(f)

    results = []
    for item in tqdm(data):
        # item should have: key, question, options, answer, llm_output
        result = xfinder_check(
            item["question"], item["options"], item["answer"], item["llm_output"]
        )
        # Attach the result to the item
        item_result = dict(item)
        item_result.update({"correct": result})
        results.append(item_result)

    # Write results to output file
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
