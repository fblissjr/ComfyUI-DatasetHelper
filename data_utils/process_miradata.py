import argparse
import json
import os
import random
from datasets import load_dataset, Dataset  # Import Dataset explicitly
from typing import Dict, List, Optional, Union
import pandas as pd

def combine_captions(example: Dict, fields_to_combine: List[str]) -> Dict:
    """Combines specified caption fields into a single 'combined_caption' field with newlines.

    Args:
        example: A dictionary representing a single row in the dataset.
        fields_to_combine: A list of field names to combine.

    Returns:
        A dictionary with the added 'combined_caption' field.
    """
    caption_fields = [ # Corrected variable name here to use caption_fields
        "dense_caption",
        "main_object_caption",
        "background_caption",
        "camera_caption",
        "style_caption",
    ]
    captions = [
        example[field] for field in caption_fields if field in example and example[field] is not None
    ]
    # Join with newlines, but remove trailing newline if any captions exist
    example["combined_caption"] = "\n".join(captions)
    return example

def save_as_csv(dataset: Dataset, output_path: str, no_header: bool):
    """Saves the dataset as a CSV file.

    Args:
        dataset: The Hugging Face dataset.
        output_path: The path to save the CSV file.
        no_header: Whether to include a header row in the CSV.
    """
    df = dataset.to_pandas()
    df.to_csv(output_path, index=False, header=(not no_header), encoding='utf-8') # Added encoding='utf-8'
    print(f"Dataset saved to CSV: {output_path}")

def save_as_hf_dataset(dataset: Dataset, output_path: str):
    """Saves the dataset in Hugging Face dataset format.

    Args:
        dataset: The Hugging Face dataset.
        output_path: The path to save the dataset.
    """
    dataset.save_to_disk(output_path)
    print(f"Dataset saved in Hugging Face format: {output_path}")

def save_as_jsonl(dataset: Dataset, output_path: str, only_combined_caption: bool):
    """Saves the dataset as a JSONL file.

    Args:
        dataset: The Hugging Face dataset.
        output_path: The path to save the JSONL file.
        only_combined_caption: Whether to output only the combined_caption field.
    """
    with open(output_path, "w", encoding='utf-8') as f: # Added encoding='utf-8'
        for example in dataset:
            if only_combined_caption:
                json.dump({"combined_caption": example["combined_caption"]}, f, ensure_ascii=False) # ensure_ascii=False
            else:
                json.dump(example, f, ensure_ascii=False) # ensure_ascii=False
            f.write("\n")
    print(f"Dataset saved to JSONL: {output_path}")

def process_and_save_dataset(
    dataset: Dataset,
    output_type: str,
    output_path: str,
    fields_to_combine: List[str],
    num_rows: Optional[int],
    row_range: Optional[List[int]],
    random_rows: bool,
    chunk_size: Optional[int],
    only_combined_caption: bool,
    no_header: bool
):
    """Processes the dataset, combines captions, and saves it to the specified format.

    Args:
        dataset: The Hugging Face dataset.
        output_type: The output format ('csv', 'hf', 'jsonl').
        output_path: The output path (file or directory).
        fields_to_combine: List of fields to combine.
        num_rows: The number of rows to process (None for all).
        row_range: A list specifying the start and end of the row range (None for not using a range).
        random_rows: Whether to select rows randomly.
        chunk_size: Chunk size for processing (None for no chunking).
        only_combined_caption: Whether to output only the combined_caption field.
        no_header: Whether to include a header row in the CSV (only for CSV output).
    """

    if row_range:
        dataset = dataset.select(range(row_range[0], row_range[1]))
    elif random_rows and num_rows:
        if num_rows > len(dataset):
            print(f"Warning: num_rows ({num_rows}) is greater than the dataset size ({len(dataset)}). Processing all rows.")
            num_rows = len(dataset)
        random_indices = random.sample(range(len(dataset)), num_rows)
        dataset = dataset.select(random_indices)
    elif num_rows:
        if num_rows > len(dataset):
            print(f"Warning: num_rows ({num_rows}) is greater than the dataset size ({len(dataset)}). Processing all rows.")
            num_rows = len(dataset)
        dataset = dataset.select(range(num_rows))

    dataset = dataset.map(lambda example: combine_captions(example, fields_to_combine))

    # Remove all columns except 'combined_caption' if only_combined_caption is True
    if only_combined_caption:
        columns_to_keep = {"combined_caption"}
        columns_to_remove = [col for col in dataset.column_names if col not in columns_to_keep]
        dataset = dataset.remove_columns(columns_to_remove)

    if chunk_size:
        if output_type in ["csv", "jsonl"]:
            num_chunks = len(dataset) // chunk_size + (len(dataset) % chunk_size != 0)
            for i in range(num_chunks):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, len(dataset))
                chunk_dataset = dataset.select(range(start, end))

                if i == 0:
                    mode = 'w'
                else:
                    mode = 'a'

                if output_type == "csv":
                    chunk_df = chunk_dataset.to_pandas()
                    chunk_df.to_csv(output_path, index=False, mode=mode, header=(not no_header) and (i==0), encoding='utf-8') # Added encoding='utf-8'
                elif output_type == "jsonl":
                    save_as_jsonl(chunk_dataset, output_path, only_combined_caption)

            print(f"Dataset saved to {output_type} in chunks: {output_path}")
        else:
            print("Chunking is only implemented for CSV and JSONL output. Saving as a single file/dataset.")
            if output_type == "csv":
                save_as_csv(dataset, output_path, no_header)
            elif output_type == "hf":
                save_as_hf_dataset(dataset, output_path)
            elif output_type == "jsonl":
                save_as_jsonl(dataset, output_path, only_combined_caption)
    else:
        if output_type == "csv":
            save_as_csv(dataset, output_path, no_header)
        elif output_type == "hf":
            save_as_hf_dataset(dataset, output_path)
        elif output_type == "jsonl":
            save_as_jsonl(dataset, output_path, only_combined_caption)

def main():
    parser = argparse.ArgumentParser(description="Combine captions in the MiraData dataset and save the result.")
    parser.add_argument(
        "--output_type",
        type=str,
        choices=["csv", "hf", "jsonl"],
        default="hf",
        help="Output format: 'csv', 'hf' (Hugging Face dataset), or 'jsonl'",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default=None,
        help="Output folder (optional). If not specified, files are saved in the current directory.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="MiraData_combined",
        help="Base output name for the resulting file(s) or directory",
    )
    parser.add_argument(
        "--fields_to_combine",
        type=str,
        nargs="*",
        default=[
            "dense_caption",
            "main_object_caption",
            "background_caption",
            "camera_caption",
            "style_caption",
        ],
        help="List of fields to combine, separated by spaces. Order determines combination order.",
    )
    parser.add_argument(
        "--num_rows",
        type=int,
        default=None,
        help="Number of rows to process (default: all rows)",
    )
    parser.add_argument(
        "--row_range",
        type=int,
        nargs=2,
        default=None,
        help="Range of rows to process (e.g., --row_range 10 100 to process rows 10 to 99).",
    )
    parser.add_argument(
        "--random_rows",
        action="store_true",
        help="Select rows randomly (to be used with --num_rows)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help="Chunk size for processing large datasets (optional)"
    )
    parser.add_argument(
        "--only_combined_caption",
        action="store_true",
        help="Output only the combined_caption field",
    )
    parser.add_argument(
        "--no_header",
        action="store_true",
        help="Do not include a header row in the CSV output (only for CSV output)",
    )

    args = parser.parse_args()

    if args.no_header and args.output_type != "csv":
        parser.error("--no_header can only be used with --output_type csv")

    # Construct output path
    if args.output_folder:
        os.makedirs(args.output_folder, exist_ok=True)
        output_path = os.path.join(args.output_folder, args.output_name)
    else:
        output_path = args.output_name

    if args.output_type == "csv":
        output_path += ".csv"
    elif args.output_type == "jsonl":
        output_path += ".jsonl"

    if args.row_range and args.random_rows:
        parser.error("--row_range and --random_rows cannot be used together.")

    dataset = load_dataset("TencentARC/MiraData", split="train")

    process_and_save_dataset(
        dataset,
        args.output_type,
        output_path,
        args.fields_to_combine,
        args.num_rows,
        args.row_range,
        args.random_rows,
        args.chunk_size,
        args.only_combined_caption,
        args.no_header
    )

if __name__ == "__main__":
    main()