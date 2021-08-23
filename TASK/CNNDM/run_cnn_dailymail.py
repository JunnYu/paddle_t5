import logging
import math
import os
import nltk
from filelock import FileLock
import paddle
from paddle.amp import GradScaler, auto_cast
from paddle.optimizer import AdamW
from paddlenlp.transformers.t5 import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm
import numpy as np
from args import parse_args
from data import get_dev_dataloader, get_train_dataloader
from utils import (
    get_writer,
    save_json,
    set_seed,
)
from datasets import load_metric as hg_load_metric
try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


logger = logging.getLogger(__name__)

@paddle.no_grad()
def evaluate(
    model, data_loader, tokenizer, metric
):
    model.eval()

    # 这个gen_kwargs是训练时候使用的，不好
    gen_kwargs = {
      "early_stopping": True,
      "length_penalty": 0.6,
      "max_length": 128,
      "min_length": 40,
      "num_beams": 4
    }

    for batch in tqdm(data_loader):
        source_ids, source_mask, labels, target_mask = batch
        generated_tokens = model.generate(
            input_ids=source_ids,
            attention_mask=source_mask,
            **gen_kwargs,
        )[0]
        labels = np.where(labels.numpy() != -100, labels.numpy(), tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(generated_tokens.numpy(), skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        metric.add_batch(predictions=decoded_preds, references=decoded_labels)

    result = metric.compute(use_stemmer=True)

    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    result = {k: round(v, 4) for k, v in result.items()}

    print(result)
    return result


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def main(args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(
                os.path.join(args.output_dir, "run.log"),
                mode="w",
                encoding="utf-8",
            )
        ],
    )
    logger.info("**********  Configuration Arguments **********")
    for arg, value in sorted(vars(args).items()):
        logger.info(f"{arg}: {value}")
    logger.info("**************************************************")
    paddle.set_device(args.device)
    set_seed(args)

    writer = get_writer(args)
    # get model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)

    # get metric
    hg_metric = hg_load_metric("rouge.py")
    # get dataloader
    train_dataloader = get_train_dataloader(tokenizer, args)
    dev_dataloader = get_dev_dataloader(tokenizer, args)

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps > 0:
        args.num_train_epochs = math.ceil(
            args.max_train_steps / num_update_steps_per_epoch
        )
    else:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    total_batch_size = args.train_batch_size * args.gradient_accumulation_steps

    decay_params = [
        p.name
        for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]

    optimizer = AdamW(
        learning_rate=args.learning_rate,
        beta1=0.9,
        beta2=0.999,
        epsilon=args.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params,
        grad_clip = paddle.nn.ClipGradByNorm(clip_norm=args.max_grad_norm)
    )

    if args.use_amp:
        scaler = GradScaler(init_loss_scaling=args.scale_loss)

    logger.info("********** Running training **********")
    logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous train batch size = {args.train_batch_size}")
    logger.info(f"  Instantaneous eval batch size = {args.eval_batch_size}")
    logger.info(f"  Total train batch size (w. accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    save_json(vars(args), os.path.join(args.output_dir, "args.json"))
    progress_bar = tqdm(range(args.max_train_steps))

    global_steps = 0
    tr_loss, logging_loss = 0.0, 0.0

    for _ in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            model.train()
            with auto_cast(
                args.use_amp, custom_white_list=["layer_norm", "softmax", "gelu"]
            ):
                source_ids, source_mask, labels, target_mask = batch
                outputs = model(
                    input_ids=source_ids,
                    attention_mask=source_mask,
                    labels=labels,
                    decoder_attention_mask=target_mask,
                )
                loss = outputs[0] / args.gradient_accumulation_steps
                tr_loss += loss.item()

            if args.use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (
                step % args.gradient_accumulation_steps == 0
                or step == len(train_dataloader) - 1
            ):
                if args.use_amp:
                    scaler.minimize(optimizer, loss)
                else:
                    optimizer.step()

                optimizer.clear_grad()
                progress_bar.update(1)
                global_steps += 1
                writer.add_scalar(
                        "train_loss",
                        loss.item(),
                        global_steps,
                    )
                if args.logging_steps > 0 and global_steps % args.logging_steps == 0 :
                    writer.add_scalar("lr", args.learning_rate, global_steps)
                    writer.add_scalar(
                        "loss",
                        (tr_loss - logging_loss) / args.logging_steps,
                        global_steps,
                    )
                    logger.info(
                        "global_steps {} - lr: {:.10f}  loss: {:.8f}".format(
                            global_steps,
                            args.learning_rate,
                            (tr_loss - logging_loss) / args.logging_steps,
                        )
                    )
                    logging_loss = tr_loss

                if args.save_steps > 0 and global_steps % args.save_steps == 0:
                    logger.info("********** Running evaluating **********")
                    logger.info(f"********** Step {global_steps} **********")
                    output_dir = os.path.join(args.output_dir, f"step-{global_steps}")
                    os.makedirs(output_dir, exist_ok=True)
                    model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    eval_results = evaluate(
                        model,
                        dev_dataloader,
                        tokenizer,
                        hg_metric
                    )
                    for k, v in eval_results.items():
                        writer.add_scalar(f"eval/{k}", v, global_steps)
                        logger.info(f"  {k} = {v}")
                    logger.info("********** Evaluating Done **********")

            if global_steps >= args.max_train_steps:
                logger.info("********** Running evaluating **********")
                logger.info(f"********** Step {global_steps} **********")
                output_dir = os.path.join(args.output_dir, f"step-{global_steps}")
                os.makedirs(output_dir, exist_ok=True)
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                eval_results = evaluate(
                    model,
                    dev_dataloader,
                    tokenizer,
                    hg_metric
                )
                for k, v in eval_results.items():
                    writer.add_scalar(f"eval/{k}", v, global_steps)
                    logger.info(f"  {k} = {v}")
                logger.info("********** Evaluating Done **********")
                logger.info("********** Training Done **********")
                return


if __name__ == "__main__":
    args = parse_args()
    main(args)
