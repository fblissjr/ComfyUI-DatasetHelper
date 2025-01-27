# Data Utilities (`./data_utils/`)

This directory contains utility scripts for working with various datasets used in ComfyUI workflows. Each subfolder typically represents a specific dataset and contains scripts to download, process, and prepare the data for use in ComfyUI nodes.

## Currently Supported Datasets

**MiraData:**  Tools for processing the MiraData dataset from TencentARC.

## General Usage

The scripts in each subfolder are typically designed to be run from the command line and may take various arguments to customize the processing. You can usually run them with the `--help` flag to see available options:

```bash
python process_miradata.py --help
```

## Dataset-Specific Instructions

### MiraData

The `process_miradata.py` script is used to process the MiraData dataset. It allows you to:

1. **Combine caption fields:** Combine multiple caption fields (e.g., `dense_caption`, `main_object_caption`, etc.) into a single `combined_caption` field. You can specify which fields to combine and the order of combination.
2. **Filter and select rows:** Process all rows, a specific number of rows, a range of rows, or randomly selected rows.
3. **Save in different formats:** Save the processed dataset as a CSV, a Hugging Face dataset, or a JSONL file.
4. **Chunk large datasets:** Process and save large datasets in chunks for efficient memory management (currently only for CSV and JSONL output).
5. **Output only combined captions** Save a file that has only the combined caption field.

**Usage Example:**

To combine the `dense_caption`, `main_object_caption`, and `style_caption` fields from the MiraData dataset and save the first 1000 rows to a Hugging Face dataset, you would run:

```bash
python process_miradata.py --output_type hf --output_folder ./mira_data_processed --output_name MiraData_Combined --num_rows 1000 --fields_to_combine dense_caption main_object_caption style_caption
```

```markdown
**Available Arguments:**

usage: process_miradata.py [-h] [--output_type {csv,hf,jsonl}] [--output_folder OUTPUT_FOLDER]
                           [--output_name OUTPUT_NAME] [--fields_to_combine FIELDS_TO_COMBINE [FIELDS_TO_COMBINE ...]]
                           [--num_rows NUM_ROWS] [--row_range ROW_RANGE ROW_RANGE] [--random_rows]
                           [--chunk_size CHUNK_SIZE] [--only_combined_caption] [--no_header]

Combine captions in the MiraData dataset and save the result.

optional arguments:
  -h, --help            show this help message and exit
  --output_type {csv,hf,jsonl}
                        Output format: 'csv', 'hf' (Hugging Face dataset), or 'jsonl' (default: hf)
  --output_folder OUTPUT_FOLDER
                        Output folder (optional). If not specified, files are saved in the current directory.
  --output_name OUTPUT_NAME
                        Base output name for the resulting file(s) or directory (default: MiraData_combined)
  --fields_to_combine FIELDS_TO_COMBINE [FIELDS_TO_COMBINE ...]
                        List of fields to combine, separated by spaces. Order determines combination order. (default:
                        ['dense_caption', 'main_object_caption', 'background_caption', 'camera_caption',
                        'style_caption'])
  --num_rows NUM_ROWS   Number of rows to process (default: all rows)
  --row_range ROW_RANGE ROW_RANGE
                        Range of rows to process (e.g., --row_range 10 100 to process rows 10 to 99).
  --random_rows         Select rows randomly (to be used with --num_rows)
  --chunk_size CHUNK_SIZE
                        Chunk size for processing large datasets (optional)
  --only_combined_caption
                        Output only the combined_caption field
  --no_header           Do not include a header row in the CSV output (only for CSV output)
```

**Primary Dependencies:**

* `datasets` (Hugging Face Datasets library)
* `pandas`
* `torch` (required by `datasets`)

**General Guidelines for New Dataset Scripts:**

* Use command-line arguments (e.g., with `argparse`) to allow users to customize the processing.
* Provide clear instructions and examples in the script's docstrings and in this README.
* Consider supporting common output formats like CSV, JSONL, and Hugging Face Datasets.
* Implement chunking or other memory management techniques for large datasets.
* Follow best practices for data processing and error handling.
