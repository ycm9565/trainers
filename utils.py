# coding = utf-8
import random
from dataclasses import dataclass, field

import torch
import numpy as np


@dataclass
class TrainingArguments:
    seed: int = 2
    train_batch_size: int = 8
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    epochs: int = 1
    print_log_per_steps: int = 10

    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})

    with_gpu: bool = field(default=True, metadata={"help": "Whether to train with gpu."})
    device_ids = [_ for _ in range(torch.cuda.device_count())]

    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})

    output_dir: str = field(default="./", metadata={
        "help": "The output directory where the model predictions and checkpoints will be written."})

    # dataloader设置
    dataloader_drop_last = False
    dataloader_num_workers = 0
    dataloader_pin_memory: bool = field(default=False,
                                        metadata={"help": "Whether or not to pin memory for DataLoader."})

    # DDP设置
    MASTER_ADDR: str = field(default="localhost", metadata={"help": "IP address."})
    MASTER_PORT: str = field(default="29500", metadata={"help": "ports."})
    backend: str = field(default="nccl", metadata={"help": "backend."})
    init_method: str = field(default="tcp", metadata={"help": "Init method,must be tcp,file or env."})
    local_rank: int = field(default=-1, metadata={"help": "Used gpu id of process."})
    world_size: int = field(default=1, metadata={"help": "The global numbers of precess."})

    # checkpoint设置
    checkpoint_dir: str = field(default="./checkpoint", metadata={"help": "The path of checkpoint."})
    save_peer_step: int = 100

    # tensorboard设置
    tensorboard_dir: str = field(default="./event", metadata={"help": "The path of tensorboard."})


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.

    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
