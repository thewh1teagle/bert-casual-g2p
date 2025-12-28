# BERT Casual G2P - Architecture Plan

## Goal
Build a Hebrew grapheme-to-phoneme (G2P) model using DictaBERT encoder with a custom phoneme decoder.

## Model Architecture

### Encoder-Decoder Setup
- **Encoder**: Pre-trained DictaBERT (`dicta-il/dictabert-large-char-menaked`)
  - Character-level tokenizer for Hebrew text
  - Frozen or fine-tunable BERT encoder
  - Produces contextual embeddings for Hebrew characters

- **Decoder**: Custom BERT decoder with cross-attention
  - Character-level phoneme tokenizer (IPA symbols)
  - 4 transformer layers with cross-attention to encoder
  - Vocab: ~85 tokens (IPA chars + English + punctuation + special tokens)
  - LM head for sequence generation

- **Framework**: HuggingFace `EncoderDecoderModel`

## Tokenizers

### Encoder Tokenizer
- Use DictaBERT's built-in character-level tokenizer
- Handles Hebrew chars with niqqud/menaked

### Decoder Tokenizer (Custom)
- Character-level IPA phoneme tokenizer
- Vocab: `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]` + IPA symbols + English letters + digits + punctuation
- Simple character splitting (no BPE/WordPiece)

## Data Format
- **Input**: Hebrew text (TSV format: `hebrew_text\tipa_phonemes`)
- **Output**: IPA phoneme sequences with stress markers (`ˈ`)
- Training data: ~13K examples
- Eval data: ~100 examples

## Code Structure

```
src/
├── config.py           # Training hyperparameters & CLI args
├── model.py            # EncoderDecoderModel creation
├── phoneme_tokenizer.py # Custom character-level IPA tokenizer
├── data.py             # TSV loading, tokenization, dataset prep
├── train.py            # Main training script
├── eval.py             # Metrics (exact match, character error rate)
├── infer.py            # Inference script
└── diagnostics.py      # Logging utilities
```

## Training Strategy
- **Loss**: Cross-entropy on phoneme sequence
- **Optimizer**: AdamW with warmup
- **Metrics**: Exact match accuracy, character error rate (CER)
- **Generation**: Beam search (num_beams=5)
- **Checkpointing**: Save best model by exact match

## Key Implementation Details
1. Separate tokenizers for encoder/decoder (critical!)
2. Decoder starts with `[CLS]`, ends with `[SEP]`
3. Label smoothing optional
4. Mixed precision training (FP16) on GPU
5. WandB integration for tracking

