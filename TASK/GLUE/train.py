import logging
import math
import os

import paddle
from paddle.amp import GradScaler, auto_cast
from paddle.optimizer import AdamW
from paddlenlp.transformers.t5 import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm

from args import parse_args
from data import get_dev_dataloader, get_train_dataloader
from metric import GLUE_METRICS
from data import GLUE_PROCESSED
from utils import (
    get_scheduler,
    get_writer,
    save_json,
    set_seed,
)


label_length_map = {
    "cola": 4,
    "sst-2": 1,
    "mrpc": 5,
    "sts-b": 5,
    "qqp": 5,
    "mnli": 4,
    "qnli": 5,
    "rte": 5,
}


logger = logging.getLogger(__name__)


@paddle.no_grad()
def evaluate(
    model, data_loader, tokenizer, label2id, metric_list, generate_max_length=5
):
    model.eval()
    all_preds = []
    all_labels = []

    for batch in data_loader:
        source_ids, source_mask, labels, target_mask = batch
        outputs = model.generate(
            input_ids=source_ids,
            attention_mask=source_mask,
            max_length=generate_max_length,
        )[0]

        for p, l, m in zip(outputs.numpy(), labels.numpy(),target_mask.numpy()):
            pred = tokenizer.decode(p, skip_special_tokens=True).strip()
            label = tokenizer.decode(l[m.astype("bool")], skip_special_tokens=True).strip()
            if label2id:
                pred = label2id[pred]
                label = label2id[label]
            else:
                pred = float(pred.replace(" ", ""))
                label = float(label.replace(" ", ""))

            all_preds.append(pred)
            all_labels.append(label)

    results = {}
    for metric in metric_list:
        results.update(metric(all_labels, all_preds))
    print(results)
    return results


def main(args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(
                os.path.join(os.path.dirname(args.output_dir), "run.log"),
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

    # metric and label
    label_name = GLUE_PROCESSED[args.task_name][1]
    if label_name:
        id2label = dict(zip(range(len(label_name)), label_name))
        label2id = dict(zip(label_name, range(len(label_name))))
    else:
        id2label = label2id = None
    metric_list = GLUE_METRICS[args.task_name]
    generate_max_length = label_length_map[args.task_name]
    writer = get_writer(args)

    # get model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)

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

    # get lr_scheduler
    lr_scheduler = get_scheduler(
        learning_rate=args.learning_rate,
        scheduler_type=args.scheduler_type,
        num_warmup_steps=args.warmup_steps
        if args.warmup_steps > 0
        else args.warmup_radio,
        num_training_steps=args.max_train_steps,
    )

    total_batch_size = args.train_batch_size * args.gradient_accumulation_steps

    decay_params = [
        p.name
        for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]

    optimizer = AdamW(
        learning_rate=lr_scheduler,
        beta1=0.9,
        beta2=0.98,
        epsilon=args.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params,
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

                lr_scheduler.step()
                optimizer.clear_grad()
                progress_bar.update(1)
                global_steps += 1

                if args.logging_steps > 0 and global_steps % args.logging_steps == 0:
                    writer.add_scalar("lr", lr_scheduler.get_lr(), global_steps)
                    writer.add_scalar(
                        "loss",
                        (tr_loss - logging_loss) / args.logging_steps,
                        global_steps,
                    )
                    logger.info(
                        "global_steps {} - lr: {:.10f}  loss: {:.8f}".format(
                            global_steps,
                            lr_scheduler.get_lr(),
                            (tr_loss - logging_loss) / args.logging_steps,
                        )
                    )
                    logging_loss = tr_loss

                if args.save_steps > 0 and global_steps % args.save_steps == 0:
                    logger.info("********** Running evaluating **********")
                    logger.info(f"********** Step {global_steps} **********")
                    output_dir = os.path.join(args.output_dir, f"step-{global_steps}")
                    os.makedirs(output_dir, exist_ok=True)

                    eval_results = evaluate(
                        model,
                        dev_dataloader,
                        tokenizer,
                        label2id,
                        metric_list,
                        generate_max_length,
                    )
                    for k, v in eval_results.items():
                        writer.add_scalar(f"eval/{k}", v, global_steps)
                        logger.info(f"  {k} = {v}")
                    model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    logger.info("********** Evaluating Done **********")

            if global_steps >= args.max_train_steps:
                logger.info("********** Training Done **********")
                return


if __name__ == "__main__":
    args = parse_args()
    main(args)
