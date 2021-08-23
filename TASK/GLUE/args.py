import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--model_name_or_path",
        default="../t5-base",
        type=str,
        help="Path to pre-trained model or shortcut name of model.",
    )
    parser.add_argument(
        "--task_name",
        default="sst-2",
        type=str,
        help="task_name.",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs",
        type=str,
        help="The output directory where the model predictions and checkpoints will be written. "
        "Default as `outputs`",
    )
    parser.add_argument(
        "--max_seq_length",
        default=256,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--train_batch_size",
        default=4,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=16,
        type=int,
        help="Batch size per GPU/CPU for evaluating.",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        default=1,
        type=int,
        help="gradient_accumulation_steps.",
    )
    parser.add_argument(
        "--learning_rate",
        default=2e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--weight_decay", default=0.01, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=4,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_train_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_radio",
        default=0.1,
        type=float,
        help="Proportion of training steps to perform linear learning rate warmup for.",
    )
    parser.add_argument("--warmup_steps", type=int, default=-1, help="warmup_steps.")
    parser.add_argument(
        "--logging_steps", type=int, default=10, help="Log every X updates steps."
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=50,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--writer_type",
        choices=["visualdl", "tensorboard"],
        default="visualdl",
        help="writer_type.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="gpu",
        help="Select which device to train model, defaults to gpu.",
    )
    parser.add_argument(
        "--scheduler_type",
        choices=["linear", "cosine", "poly"],
        default="linear",
        type=str,
        help="scheduler_type.",
    )
    parser.add_argument(
        "--use_amp", action="store_true", help="Enable mixed precision training."
    )
    parser.add_argument(
        "--is_test", action="store_true", help="is_test."
    )
    parser.add_argument(
        "--scale_loss",
        type=float,
        default=2 ** 15,
        help="The value of scale_loss for fp16.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="num_workers.",
    )
    args = parser.parse_args()
    args.task_name = args.task_name.lower()
    args.logdir = os.path.join(args.output_dir, "logs")
    os.makedirs("caches", exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)

    return args
