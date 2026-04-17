from datasets import load_dataset, load_from_disk
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", required=True, type=str)
parser.add_argument("--chosen_aggregated_jsonl_path", required=True, type=str)
parser.add_argument("--rejected_aggregated_jsonl_path", required=True, type=str)
parser.add_argument("--save_dataset_path", required=True, type=str)

args = parser.parse_args()

meta_data = load_from_disk(args.dataset_path)
# remove all chosen/rejected columns
meta_data = meta_data.remove_columns(
    [
        col
        for col in meta_data.column_names
        if col.startswith("chosen") or col.startswith("rejected")
    ]
)

for mode, path in zip(
    ["chosen", "rejected"],
    [args.chosen_aggregated_jsonl_path, args.rejected_aggregated_jsonl_path],
):

    phi4_data = load_dataset(
        "json",
        data_files=path,
        split="train",
    )
    print("Raw length:", len(phi4_data))
    phi4_data = phi4_data.filter(lambda x: x, input_columns=[mode])
    print("Filtered length:", len(phi4_data))

    if mode == "chosen":
        phi4_data = phi4_data.remove_columns(["rejected"])
    else:
        phi4_data = phi4_data.remove_columns(["chosen"])

    def map_result_back(phi4_data, meta_data):
        phi4_data_dict = {}
        for item in phi4_data:
            phi4_data_dict[item["id_string"]] = item

        meta_data_matched = meta_data.filter(
            lambda x: x in phi4_data_dict, input_columns=["id_string"]
        )

        def process(item):
            # find the target one and then add the feature
            return phi4_data_dict[item["id_string"]]

        meta_data_matched = meta_data_matched.map(
            process, writer_batch_size=200, num_proc=8
        )

        print(
            f"Removed: {len(meta_data)-len(meta_data_matched)}. Final: {len(meta_data_matched)}"
        )
        return meta_data_matched

    meta_data = map_result_back(phi4_data, meta_data)  # merge data

meta_data.save_to_disk(args.save_dataset_path)
print("Saved to", args.save_dataset_path)
