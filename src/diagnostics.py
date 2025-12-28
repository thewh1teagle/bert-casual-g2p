def log_trainable_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {trainable:,} / {total:,} params ({100*trainable/total:.1f}% trainable, {trainable*4/1024**3:.2f} GB)")


def log_encoder_decoder_params(model):
    if hasattr(model, 'encoder') and hasattr(model, 'decoder'):
        enc = sum(p.numel() for p in model.encoder.parameters())
        dec = sum(p.numel() for p in model.decoder.parameters())
        total = sum(p.numel() for p in model.parameters())
        print(f"Encoder: {enc:,} ({100*enc/total:.1f}%) | Decoder: {dec:,} ({100*dec/total:.1f}%)")


def log_samples(dataset, encoder_tokenizer, decoder_tokenizer, split_name: str = "Train", num_samples: int = 2):
    print(f"\n{split_name} samples:")
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        inp = encoder_tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
        labels = [l if l != -100 else decoder_tokenizer.pad_token_id for l in sample['labels']]
        tgt = decoder_tokenizer.decode(labels, skip_special_tokens=True)
        print(f"  [{i+1}] {inp} → {tgt}")


def log_config(config):
    print(f"\nConfig: {config.model_name} | {config.num_epochs} epochs, bs={config.batch_size}, lr={config.learning_rate}")
    print(f"Data: {config.data_file} | Output: {config.output_dir}")
