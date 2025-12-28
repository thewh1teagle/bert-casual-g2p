"""Inference with separate encoder/decoder tokenizers."""

import argparse
import torch
from transformers import EncoderDecoderModel, AutoTokenizer
from typing import List
from phoneme_tokenizer import create_phoneme_tokenizer


def load_model(checkpoint_path: str, device: str = None):
    """Load model and both tokenizers from checkpoint."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = EncoderDecoderModel.from_pretrained(checkpoint_path)
    encoder_tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    decoder_tokenizer = create_phoneme_tokenizer()

    return model.to(device).eval(), encoder_tokenizer, decoder_tokenizer, device


def predict(model, encoder_tokenizer, decoder_tokenizer, text: str, max_length: int = 128, num_beams: int = 4, device: str = "cpu") -> str:
    """Generate phoneme prediction for input text."""
    inputs = encoder_tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length, num_beams=num_beams, early_stopping=True, no_repeat_ngram_size=3)

    return decoder_tokenizer.decode(outputs[0], skip_special_tokens=True)


def predict_batch(model, encoder_tokenizer, decoder_tokenizer, texts: List[str], max_length: int = 128, num_beams: int = 4, device: str = "cpu") -> List[str]:
    """Generate phoneme predictions for batch of texts."""
    inputs = encoder_tokenizer(texts, return_tensors="pt", max_length=max_length, truncation=True, padding=True).to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length, num_beams=num_beams, early_stopping=True, no_repeat_ngram_size=3)

    return decoder_tokenizer.batch_decode(outputs, skip_special_tokens=True)


def interactive_mode(model, encoder_tokenizer, decoder_tokenizer, device: str, num_beams: int = 4):
    """Run interactive G2P conversion."""
    print(f"\n{'='*60}\nInteractive G2P Mode\n{'='*60}")
    print("Enter text to convert (or 'quit' to exit)\n")

    while True:
        try:
            text = input("Input: ").strip()
            if not text or text.lower() in ["quit", "exit", "q"]:
                break
            prediction = predict(model, encoder_tokenizer, decoder_tokenizer, text, num_beams=num_beams, device=device)
            print(f"Output: {prediction}\n")
        except (KeyboardInterrupt, Exception) as e:
            if isinstance(e, KeyboardInterrupt):
                print("\n\nExiting...")
            else:
                print(f"Error: {e}\n")
            break


def main():
    parser = argparse.ArgumentParser(description="G2P inference with trained model")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--batch", type=str, nargs="+", default=None)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    model, encoder_tokenizer, decoder_tokenizer, device = load_model(args.model_path, args.device)

    if args.text:
        print(f"\nInput:  {args.text}")
        prediction = predict(model, encoder_tokenizer, decoder_tokenizer, args.text, args.max_length, args.num_beams, device)
        print(f"Output: {prediction}\n")
    elif args.batch:
        predictions = predict_batch(model, encoder_tokenizer, decoder_tokenizer, args.batch, args.max_length, args.num_beams, device)
        print("\nResults:")
        for text, pred in zip(args.batch, predictions):
            print(f"  {text} → {pred}")
        print()
    else:
        interactive_mode(model, encoder_tokenizer, decoder_tokenizer, device, args.num_beams)


if __name__ == "__main__":
    main()
