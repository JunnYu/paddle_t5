import argparse
import logging

import datasets
import nltk
import torch
import transformers
from accelerate import Accelerator
from datasets import Dataset, load_metric
from filelock import FileLock
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    DataCollatorForSeq2Seq,
    T5ForConditionalGeneration,
    T5TokenizerFast,
)
from transformers.file_utils import is_offline_mode

logger = logging.getLogger(__name__)


try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate T5 model on a summarize task"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="../step-25000-aistudio-20.35",
        help="model_name_or_path",
    )
    parser.add_argument(
        "--evaluate_file",
        type=str,
        default="cnn_dailymail_dev.json",
        help="evaluate_file.",
    )
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to "
        "padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=1024,
        help="The maximum total input sequence length after "
        "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--source_prefix",
        type=str,
        default="summarize: ",
        help="A prefix to add before every source text " "(useful for T5 models).",
    )
    parser.add_argument(
        "--overwrite_cache",
        type=bool,
        default=False,
        help="Overwrite the cached evaluation sets",
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=512,
        help="The maximum total sequence length for target text after "
        "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        "during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
        help="Number of beams to use for evaluation. This argument will be "
        "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--length_penalty",
        type=float,
        default=0.6,
        help="length_penalty",
    )
    parser.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=None,
        help="no_repeat_ngram_size",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=24,
        help="Batch size (per device) for the evaluation dataloader.",
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    accelerator = Accelerator()
    logger.info(accelerator.state)
    logger.setLevel(
        logging.INFO if accelerator.is_local_main_process else logging.ERROR
    )
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    tokenizer = T5TokenizerFast.from_pretrained("t5-base")
    raw_datasets = Dataset.from_json(args.evaluate_file)

    prefix = args.source_prefix if args.source_prefix is not None else ""
    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        inputs = examples["article"]
        targets = examples["highlights"]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(
            inputs, max_length=args.max_source_length, padding=padding, truncation=True
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=args.max_target_length,
                padding=padding,
                truncation=True,
            )

        if padding == "max_length" and args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label]
                for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def postprocess_text(datas):
        datas = [data.strip() for data in datas]
        datas = ["\n".join(nltk.sent_tokenize(data)) for data in datas]
        return datas

    NOT_TRUNC_LABELS = postprocess_text(raw_datasets["highlights"])

    eval_dataset = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=["article", "highlights", "id"],
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )

    label_pad_token_id = (
        -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=label_pad_token_id
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )

    metric = load_metric("rouge.py")

    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)
    ######################################################################
    gen_kwargs = {
        "max_length": args.max_target_length,
        "min_length": 50,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "num_beams": args.num_beams,
        "length_penalty": args.length_penalty,
        "early_stopping": True,
    }
    #####################################################################
    model.eval()

    all_preds = []
    with torch.no_grad():
        for batch in tqdm(eval_dataloader):
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **gen_kwargs,
            )
            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()

            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            decoded_preds = tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )
            decoded_preds = postprocess_text(decoded_preds)
            all_preds.extend(decoded_preds)

    for pred, label in zip(all_preds, NOT_TRUNC_LABELS):
        metric.add(prediction=pred, reference=label)

    result = metric.compute(use_stemmer=True)

    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    result = {k: round(v, 4) for k, v in result.items()}

    logger.info(result)
    torch.save(all_preds, "all_preds.pt")


if __name__ == "__main__":
    main()
