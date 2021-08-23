import json
import pickle
import random


import numpy as np
import paddle

from paddlenlp.transformers import (
    CosineDecayWithWarmup,
    LinearDecayWithWarmup,
    PolyDecayWithWarmup,
)

scheduler_type2cls = {
    "linear": LinearDecayWithWarmup,
    "cosine": CosineDecayWithWarmup,
    "poly": PolyDecayWithWarmup,
}



def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)


def get_writer(args):
    if args.writer_type == "visualdl":
        from visualdl import LogWriter

        writer = LogWriter(logdir=args.logdir)
    elif args.writer_type == "tensorboard":
        from tensorboardX import SummaryWriter

        writer = SummaryWriter(logdir=args.logdir)
    else:
        raise ValueError("writer_type must be in ['visualdl', 'tensorboard']")
    return writer


def get_scheduler(
    learning_rate,
    scheduler_type,
    num_warmup_steps=None,
    num_training_steps=None,
    **scheduler_kwargs,
):
    if scheduler_type not in scheduler_type2cls.keys():
        data = " ".join(scheduler_type2cls.keys())
        raise ValueError(f"scheduler_type must be choson from {data}")

    if num_warmup_steps is None:
        raise ValueError(f"requires `num_warmup_steps`, please provide that argument.")

    if num_training_steps is None:
        raise ValueError(
            f"requires `num_training_steps`, please provide that argument."
        )

    return scheduler_type2cls[scheduler_type](
        learning_rate=learning_rate,
        total_steps=num_training_steps,
        warmup=num_warmup_steps,
        **scheduler_kwargs,
    )


def save_json(data, file_name):
    with open(file_name, "w", encoding="utf-8") as w:
        w.write(json.dumps(data, ensure_ascii=False, indent=4) + "\n")



def save_pickle(data, file_path):
    with open(str(file_path), "wb") as f:
        pickle.dump(data, f)


def load_pickle(input_file):
    with open(str(input_file), "rb") as f:
        data = pickle.load(f)
    return data
