"""Character-level tokenizer for IPA phonemes and mixed content."""

from typing import List, Dict, Optional
from transformers import PreTrainedTokenizer


class PhonemeTokenizer(PreTrainedTokenizer):
    """Character-level tokenizer with 85-token vocabulary (IPA + English + digits + punctuation)."""

    def __init__(self, unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]", **kwargs):
        # Build vocabulary first
        special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]']
        ipa_chars = ['ɡ', 'ʁ', 'ʃ', 'ʒ', 'ʔ', 'χ', 'ˈ']
        lowercase = list('abcdefghijklmnopqrstuvwxyz')
        uppercase = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        digits = list('0123456789')
        punctuation = [' ', '!', '"', "'", ',', '-', '.', ':', ';', '?', '(', ')']

        all_chars = sorted(set(ipa_chars + lowercase + uppercase + digits + punctuation))
        all_tokens = special_tokens + all_chars

        self.encoder = {token: idx for idx, token in enumerate(all_tokens)}
        self.decoder = {idx: token for token, idx in self.encoder.items()}
        self._vocab_size = len(self.encoder)

        super().__init__(unk_token=unk_token, sep_token=sep_token, pad_token=pad_token, cls_token=cls_token, **kwargs)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def get_vocab(self) -> Dict[str, int]:
        return self.encoder.copy()

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        """Split text into characters."""
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self.encoder.get(token, self.encoder['[UNK]'])

    def _convert_id_to_token(self, index: int) -> str:
        return self.decoder.get(index, '[UNK]')

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Join characters into string."""
        return ''.join(tokens)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple:
        """Save vocabulary to JSON file."""
        import os
        import json

        os.makedirs(save_directory, exist_ok=True)
        vocab_file = os.path.join(save_directory, (filename_prefix + "-" if filename_prefix else "") + "vocab.json")

        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self.encoder, f, ensure_ascii=False, indent=2)

        return (vocab_file,)

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        """Add [CLS] and [SEP] tokens."""
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        return [self.cls_token_id] + token_ids_0 + [self.sep_token_id] + token_ids_1 + [self.sep_token_id]

    def get_special_tokens_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False) -> List[int]:
        """Return mask indicating special tokens (1) vs regular tokens (0)."""
        if already_has_special_tokens:
            return super().get_special_tokens_mask(token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True)

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        """Create token type IDs (all 0s for single sequence)."""
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]


def create_phoneme_tokenizer() -> PhonemeTokenizer:
    """Create character-level phoneme tokenizer."""
    return PhonemeTokenizer()
