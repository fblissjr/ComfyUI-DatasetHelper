import datetime
import json
import os
import random
from typing import Dict, List, Tuple, Optional

import torch
from comfy.cli_args import args
import pandas as pd
import folder_paths  # Import folder_paths
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from .utils import log # Import log
import logging

from server import PromptServer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

# --- Configuration (Adjust these in your workflow or node inputs) ---
OUTPUT_FOLDER = folder_paths.get_output_directory()  # Get ComfyUI output directory
METADATA_SUBFOLDER = "dataset_batch_metadata" # Subfolder for metadata within the output directory
# --- End Configuration ---

class DatasetBatchNode:
    """
    Custom ComfyUI node to handle batch processing of a dataset.
    """

    def __init__(self):
        self.output_dir = os.path.join(OUTPUT_FOLDER, METADATA_SUBFOLDER) # Metadata subfolder path
        self.type = "output"
        self.metadata = []
        self.current_row_index = 0
        self.rows_to_process = []
        self.prompts = []
        self.total_rows = 0  # Track the total number of rows in the dataset
        self.dataset_length = 0
        self.dataset = None  # Store the loaded dataset
        self.magic_number = 0 # Initialise magic number

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_path": ("STRING", {"default": ""}),
                "prompt_field": ("STRING", {"default": "text"}),
                "num_rows": ("INT", {"default": -1, "min": -1, "max": 10000, "step": 1}),  # -1 for all rows
                "start_row": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "random_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "shuffle": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "delimiter": ("STRING", {"default": "\\n"}),
                "magic_number": ("INT", {"default": 0, "min": 0, "max": 9999999, "forceInput": True}),
                "text_input": ("STRING", {"forceInput": True, "default": ""}),
                "mixed_fields_config_json": ("STRING", {"multiline": True, "default": '[\n    {\n        "field": "dense_caption",\n        "filter": ""\n    },\n    {\n        "field": "style_caption",\n        "filter": ""\n    }\n]'})
            }
        }

    RETURN_TYPES = ("STRING", "INT",)
    RETURN_NAMES = ("prompt", "seed",)
    FUNCTION = "process_batch"
    CATEGORY = "Dataset"
    OUTPUT_NODE = True

    def process_batch(
        self,
        dataset_path: str,
        prompt_field: str,
        num_rows: int,
        start_row: int,
        random_seed: int,
        shuffle: bool,
        magic_number: int = 0,
        delimiter: str = "\\n",
        text_input: str = "",
        mixed_fields_config_json: str = None,
        **kwargs,
    ) -> Tuple[str, int]:
        from datasets import load_dataset

        # Load the dataset only once
        if self.dataset is None:
            try:
                self.dataset = self.load_dataset(dataset_path)
            except Exception as e:
                log.error(f"Error loading dataset: {e}")
                return ("", 0,)

            # Check if the specified prompt field exists
            if prompt_field not in self.dataset.column_names:
                raise ValueError(f"Prompt field '{prompt_field}' not found in dataset. Available fields: {self.dataset.column_names}")

            # Determine the rows to process
            if num_rows == -1:
                num_rows = len(self.dataset)

            self.dataset_length = len(self.dataset)
            effective_start_row = start_row + self.magic_number

            end_row = min(effective_start_row + num_rows, self.dataset_length)
            
            if shuffle:
                random.seed(random_seed)
                shuffled_indices = list(range(len(self.dataset)))
                random.shuffle(shuffled_indices)
                self.rows_to_process = shuffled_indices[effective_start_row:end_row]
            else:
                self.rows_to_process = list(range(effective_start_row, end_row))

        log.debug(f"DatasetBatchNode: process_batch called. Current magic_number: {self.magic_number}, start_row: {start_row}, num_rows: {num_rows}")

        # Only process if there are rows left to process
        if self.current_row_index < len(self.rows_to_process):
            # Process the current row
            current_row_index = self.rows_to_process[self.current_row_index]
            row = self.dataset[current_row_index]
            seed = random.randint(0, 2**32 - 1)
            combined_prompt_segments = []

            if mixed_fields_config_json:
                # Parse mixed fields config from JSON string
                try:
                    mixed_fields_config = json.loads(mixed_fields_config_json)
                    if not isinstance(mixed_fields_config, list):
                        raise ValueError("mixed_fields_config must be a JSON list of field configurations.")
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in mixed_fields_config: {e}")

                # Process each field config
                for field_config in mixed_fields_config:
                    field_name = field_config.get("field")
                    filter_query = field_config.get("filter", "")

                    if not field_name:
                        log.warning("Skipping field config due to missing 'field' name.")
                        continue

                    if filter_query.strip():
                        # Apply filter
                        filtered_dataset = self.dataset.filter(lambda example: eval(filter_query, {}, {"example": example}))
                        if not filtered_dataset:
                            log.warning(f"No rows found matching filter: '{filter_query}' for field '{field_name}'. Using current row instead.")
                            selected_row = row
                        else:
                            selected_row = random.choice(filtered_dataset)
                    else:
                        selected_row = row

                    caption_segment = selected_row.get(field_name, "")
                    combined_prompt_segments.append(caption_segment)

                # Combine segments with newlines
                prompt_text = delimiter.join([segment.strip() for segment in combined_prompt_segments if segment])

            else:
                prompt_text = row[prompt_field]

            # Combine with text_input if provided
            if text_input:
                prompt_text = text_input + delimiter + prompt_text

            # Trim whitespace from the start and end of the prompt
            prompt_text = prompt_text.strip()

            # Metadata entry
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            metadata_entry = {
                "row_index": current_row_index,
                "seed": seed,
                "magic_number": self.magic_number,
                "output_filename": f"row_{current_row_index}_{seed}_{timestamp}",
                "prompt": prompt_text,
                "mixed_fields_config": mixed_fields_config_json,
                **row,
            }

            self.metadata.append(metadata_entry)

            log.info(f"Processing row {current_row_index} with seed {seed}...")

            # Save metadata to a JSONL file after processing each row
            os.makedirs(self.output_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            metadata_filename = os.path.join(self.output_dir, f"metadata_{timestamp}.jsonl")
            with open(metadata_filename, "a", encoding='utf-8') as f:
                json.dump(metadata_entry, f, ensure_ascii=False)
                f.write("\n")
            log.info(f"Metadata for row {current_row_index} saved to {metadata_filename}")
            
            # Increment the row index and magic_number
            self.current_row_index += 1
            self.magic_number += 1  # Increment magic_number for the next row

            # Dispatch custom event after processing each row
            PromptServer.instance.send_sync("dataset_row_processed", {
                "node_id": id(self),  # Pass the unique ID of the node instance
                "magic_number": self.magic_number,  # Pass the updated magic_number
                "row_index": current_row_index,
            })

            # Check if all rows for this batch are processed
            if self.current_row_index >= len(self.rows_to_process):
                self.current_row_index = 0  # Reset for the next batch
                self.rows_to_process = []  # Clear the processed rows
                log.info(f"DatasetBatchNode: Batch complete. magic_number updated to {self.magic_number}")

            return (prompt_text, seed,)
        else:
            # If there are no more rows to process, you might want to handle this case.
            # For example, you can reset the magic_number or log a message.
            log.info("Dataset processing complete.")
            self.magic_number = 0  # Reset magic_number
            self.current_row_index = 0  # Reset current_row_index
            self.rows_to_process = []  # Clear the processed rows
            return ("", 0)  # Indicate end of processing

    def load_dataset(self, dataset_path: str) -> Dataset:
        """Loads a dataset from a file (CSV, JSONL) or a Hugging Face dataset."""
        from datasets import load_dataset, load_from_disk

        if os.path.isdir(dataset_path):
            # Load from disk
            return load_from_disk(dataset_path)
        elif os.path.isfile(dataset_path):
            # Load from a local file
            file_ext = os.path.splitext(dataset_path)[1].lower()
            if file_ext == ".jsonl":
                data = []
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            log.error(f"Error decoding JSON line: {e}")
                            continue
                return Dataset.from_list(data)
            elif file_ext == ".csv":
                df = pd.read_csv(dataset_path)
                return Dataset.from_pandas(df)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
        else:
            # Assume it's a Hugging Face dataset path
            try:
                # Try to load the dataset with a 'train' split
                return load_dataset(dataset_path, split="train")
            except ValueError:
                # If 'train' split not found, try loading without specifying the split
                return load_dataset(dataset_path, split=None)

WEB_DIRECTORY = "./js"

NODE_CLASS_MAPPINGS = {
    "DatasetBatchNode": DatasetBatchNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DatasetBatchNode": "Dataset Batch Node"
}