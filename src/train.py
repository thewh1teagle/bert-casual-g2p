import torch
import os
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from config import get_config, set_random_seeds, MAX_LENGTH
from model import create_model
from data import load_tsv_data, prepare_dataset, split_dataset, create_data_collator
from eval import create_compute_metrics
from diagnostics import log_trainable_params, log_encoder_decoder_params, log_samples, log_config


def main():
    config = get_config()
    set_random_seeds(config.seed)
    log_config(config)

    train_df = load_tsv_data(config.data_file)
    eval_df = load_tsv_data(config.eval_file) if config.eval_file else None

    model, encoder_tokenizer, decoder_tokenizer = (
        create_model(from_pretrained_path=config.resume_from_checkpoint)
        if config.resume_from_checkpoint
        else create_model(encoder_model_name=config.model_name, decoder_layers=config.decoder_layers)
    )
    log_trainable_params(model)
    log_encoder_decoder_params(model)

    os.makedirs(config.cache_dir, exist_ok=True)

    if eval_df is not None:
        train_dataset = prepare_dataset(train_df, encoder_tokenizer, decoder_tokenizer, MAX_LENGTH, os.path.join(config.cache_dir, "train.arrow"))
        val_dataset = prepare_dataset(eval_df, encoder_tokenizer, decoder_tokenizer, MAX_LENGTH, os.path.join(config.cache_dir, "eval.arrow"))
    else:
        full_dataset = prepare_dataset(train_df, encoder_tokenizer, decoder_tokenizer, MAX_LENGTH, os.path.join(config.cache_dir, "full.arrow"))
        train_dataset, val_dataset = split_dataset(full_dataset, train_ratio=0.8, seed=config.seed)

    print(f"Dataset: {len(train_dataset)} train, {len(val_dataset)} val")
    log_samples(train_dataset, encoder_tokenizer, decoder_tokenizer, "Train", 2)
    log_samples(val_dataset, encoder_tokenizer, decoder_tokenizer, "Val", 2)

    data_collator = create_data_collator(encoder_tokenizer, model)
    compute_metrics = create_compute_metrics(decoder_tokenizer)

    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_grad_norm=config.max_grad_norm,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=(config.metric_for_best_model == "exact_match"),
        logging_steps=config.logging_steps,
        report_to=config.report_to.split(",") if config.report_to != "none" else [],
        predict_with_generate=True,
        generation_max_length=config.generation_max_length,
        generation_num_beams=config.num_beams,
        seed=config.seed,
        data_seed=config.seed,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        remove_unused_columns=True,
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=encoder_tokenizer,
    )

    print("\n" + "="*60)
    print("Starting Training")
    print("="*60 + "\n")

    try:
        trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving...")
        trainer.save_model(os.path.join(config.output_dir, "interrupted"))
        return

    print(f"\n\nTraining complete! Saving to {config.output_dir}...")
    trainer.save_model()
    encoder_tokenizer.save_pretrained(config.output_dir)
    print(f"Inference: uv run src/infer.py --model_path {config.output_dir} --text 'שלום'\n")


if __name__ == "__main__":
    main()
