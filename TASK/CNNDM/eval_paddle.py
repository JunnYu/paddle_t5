import nltk
import paddle
from filelock import FileLock
from datasets import Dataset, load_metric
from data import get_dev_dataloader
from tqdm.auto import tqdm
from paddlenlp.transformers.t5 import T5ForConditionalGeneration,T5Tokenizer
import argparse

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate T5 model on a summarize task"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="step-25000-aistudio-20.35",
        help="model_name_or_path",
    )
    parser.add_argument(
        "--evaluate_file",
        type=str,
        default="./caches/cnndailymail/cnn_dailymail_dev.json",
        help="evaluate_file.",
    )
    parser.add_argument(
        "--source_prefix",
        type=str,
        default="summarize: ",
        help="A prefix to add before every source text " "(useful for T5 models).",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=1024,
        help="The maximum total input sequence length after "
        "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
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
        default=1.0,
        help="length_penalty",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=24,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="num_workers.",
    )
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    def postprocess_text(datas):
        datas = [data.strip() for data in datas]
        datas = ["\n".join(nltk.sent_tokenize(data)) for data in datas]
        return datas

    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    model.eval()

    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    eval_dataloader = get_dev_dataloader(tokenizer, args)
    raw_datasets = Dataset.from_json(args.evaluate_file)
    NOT_TRUNC_LABELS = postprocess_text(raw_datasets["highlights"])
    metric = load_metric("rouge.py")

    gen_kwargs = {
        "max_length": args.max_target_length - 1,
        "min_length": 50,
        "num_beams": args.num_beams,
        "length_penalty": args.length_penalty,
        "early_stopping": True,
        "decode_strategy": "beam_search"
    }

    all_preds = []
    with paddle.no_grad():
        for batch in tqdm(eval_dataloader):
            source_ids, source_mask, labels, target_mask = batch
            generated_tokens = model.generate(
                input_ids=source_ids,
                attention_mask=source_mask,
                **gen_kwargs,
            )[0]
            decoded_preds = tokenizer.batch_decode(
                generated_tokens.numpy(), skip_special_tokens=True
            )
            decoded_preds = postprocess_text(decoded_preds)
            all_preds.extend(decoded_preds)

    for pred, label in zip(all_preds, NOT_TRUNC_LABELS):
        metric.add(prediction=pred, reference=label)

    result = metric.compute(use_stemmer=True)

    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    result = {k: round(v, 4) for k, v in result.items()}

    print(result)
    paddle.save(all_preds, "all_preds.pd")

if __name__ == "__main__":
    main()
