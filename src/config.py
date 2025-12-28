import argparse
import os
import random
import numpy as np
import torch
from transformers import set_seed

MAX_LENGTH = 128


def get_config():
    parser = argparse.ArgumentParser(description="Train DictaBERT encoder-decoder for Hebrew G2P")

    parser.add_argument("--data_file", type=str, default="data/data.tsv")
    parser.add_argument("--eval_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--cache_dir", type=str, default=".cache")

    parser.add_argument("--model_name", type=str, default="dicta-il/dictabert-large-char-menaked")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--decoder_layers", type=int, default=4)

    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--metric_for_best_model", type=str, default="wer")

    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--generation_max_length", type=int, default=MAX_LENGTH)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--wandb_mode", type=str, default="online")
    parser.add_argument("--wandb_project", type=str, default="dictabert-g2p")

    args = parser.parse_args()
    
    # Set wandb environment variables
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode
        if args.wandb_project:
            os.environ["WANDB_PROJECT"] = args.wandb_project
    
    return args


def set_random_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)
