"""Model creation with separate encoder and decoder tokenizers."""

from transformers import AutoTokenizer, AutoModel, EncoderDecoderModel, BertConfig, BertLMHeadModel
from phoneme_tokenizer import create_phoneme_tokenizer


def create_model(
    encoder_model_name: str = "dicta-il/dictabert-large-char-menaked",
    decoder_layers: int = 4,
    from_pretrained_path: str = None,
):
    """Create encoder-decoder model with separate tokenizers for encoder (DictaBERT) and decoder (phonemes)."""
    if from_pretrained_path:
        model = EncoderDecoderModel.from_pretrained(from_pretrained_path)
        encoder_tokenizer = AutoTokenizer.from_pretrained(from_pretrained_path)
        decoder_tokenizer = create_phoneme_tokenizer()
        return model, encoder_tokenizer, decoder_tokenizer

    # Encoder: DictaBERT (pretrained)
    encoder = AutoModel.from_pretrained(encoder_model_name)
    encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_model_name)
    if encoder_tokenizer.pad_token is None:
        encoder_tokenizer.pad_token = encoder_tokenizer.eos_token

    # Decoder: Custom phoneme tokenizer
    decoder_tokenizer = create_phoneme_tokenizer()

    # Decoder: BERT with cross-attention
    decoder_config = BertConfig(
        vocab_size=len(decoder_tokenizer),
        hidden_size=encoder.config.hidden_size,
        num_hidden_layers=decoder_layers,
        num_attention_heads=encoder.config.num_attention_heads,
        intermediate_size=encoder.config.intermediate_size,
        max_position_embeddings=128,
        is_decoder=True,
        add_cross_attention=True,
    )

    decoder = BertLMHeadModel(decoder_config)
    model = EncoderDecoderModel(encoder=encoder, decoder=decoder)

    # Configure generation tokens
    model.config.decoder_start_token_id = decoder_tokenizer.cls_token_id
    model.config.eos_token_id = decoder_tokenizer.sep_token_id
    model.config.pad_token_id = decoder_tokenizer.pad_token_id
    model.config.bos_token_id = decoder_tokenizer.cls_token_id

    # Also set in generation config
    model.generation_config.decoder_start_token_id = decoder_tokenizer.cls_token_id
    model.generation_config.eos_token_id = decoder_tokenizer.sep_token_id
    model.generation_config.pad_token_id = decoder_tokenizer.pad_token_id
    model.generation_config.bos_token_id = decoder_tokenizer.cls_token_id

    return model, encoder_tokenizer, decoder_tokenizer
