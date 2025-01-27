# ComfyUI Dataset Batch Node

This custom node set for ComfyUI provides a `DatasetBatchNode` for automated, sequential processing of datasets, particularly useful for iterative training or batched image/video generation workflows.

**Key Features:**

*   Loads datasets from local files (JSONL, CSV) or Hugging Face `datasets`.
*   Processes individual rows sequentially, triggering a new job for each.
*   `magic_number` input automatically tracks the current row index across multiple jobs, ensuring no rows are missed or duplicated, even if you stop and restart the workflow.
*   Supports a flexible `mixed_fields_config_json` input to define complex prompt construction from multiple dataset fields and filters.
*   Includes a script (`process_miradata.py`) to prepare datasets for use with the node.

**Node Configuration (`DatasetBatchNode`)**

*   **`dataset_path`:** (Required) Path to your dataset file (JSONL, CSV) or the Hugging Face dataset name.
*   **`prompt_field`:** (Required) The field in your dataset containing the base prompt text.
*   **`num_rows`:** (Required) The number of rows to process (-1 for all rows). The node will stop after processing this number of rows
*   **`start_row`:** (Required) The starting row index (0-based).
*   **`random_seed`:** (Required) Seed for shuffling.
*   **`shuffle`:** (Required) Whether to shuffle the rows before processing.
*   **`magic_number`:** (Optional) An automatically managed offset that ensures sequential processing across multiple workflow runs. **Do not modify this manually.** Let the node and extension manage it.
*   **`delimiter`:** (Optional) Delimiter to use when combining multiple prompt fields (default: `\n`).
*   **`text_input`:** (Optional) Text to append before the prompt.
*   **`mixed_fields_config_json`:** (Optional) A JSON string defining how to combine multiple fields from your dataset into a single prompt. See the "Advanced Prompt Construction" section below.

**Important:** The `magic_number` input is used for processing each row. There might be a smarter way to do this in Comfy but that's where I landed for now. The JS extension (`dataset_batch_automation.js`) handles the automatic incrementing of this value to ensure proper batch progression.

**Advanced Prompt Construction (`mixed_fields_config_json`)**

This optional input allows you to create complex prompts by combining multiple fields from your dataset, potentially applying filters to each field.

**Format:**

```json
[
    {
        "field": "field_name_1",
        "filter": "filter_expression_1"
    },
    {
        "field": "field_name_2",
        "filter": "filter_expression_2"
    }
]

[
    {
        "field": "description",
        "filter": "len(example['description']) > 20"
    },
    {
        "field": "tags",
        "filter": "'cat' in example['tags']"
    }
]```



