"""Data loading and preprocessing with separate encoder/decoder tokenizers."""

from pathlib import Path
import pandas as pd
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq


def load_tsv_data(file_path: str) -> pd.DataFrame:
    """Load TSV data with tab-separated input/output columns."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    rows = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 2:
                rows.append({"input": parts[0], "output": parts[1]})

    return pd.DataFrame(rows)


def prepare_dataset(df: pd.DataFrame, encoder_tokenizer, decoder_tokenizer, max_length: int = 128, cache_file: str = None):
    """Tokenize data using encoder tokenizer for inputs and decoder tokenizer for outputs."""
    def tokenize_function(examples):
        # Encode inputs with encoder tokenizer (DictaBERT)
        model_inputs = encoder_tokenizer(examples["input"], max_length=max_length, truncation=True)

        # Encode labels with decoder tokenizer (phoneme)
        labels = decoder_tokenizer(examples["output"], max_length=max_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    dataset = Dataset.from_pandas(df)
    return dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        cache_file_name=cache_file,
    )


def split_dataset(dataset, train_ratio: float = 0.8, seed: int = 42):
    """Split dataset into train and validation sets."""
    dataset = dataset.shuffle(seed=seed)
    train_size = int(train_ratio * len(dataset))
    return dataset.select(range(train_size)), dataset.select(range(train_size, len(dataset)))


def create_data_collator(tokenizer, model):
    """Create data collator for dynamic padding."""
    return DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=-100)
