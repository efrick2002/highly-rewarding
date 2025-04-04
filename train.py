import argparse
import os
import yaml
import json
import random
from transformers import Trainer, TrainingArguments, set_seed, AutoModelForCausalLM
from losses import REGISTERED_LOSSES
from typing import Optional
from torch.utils.data import Sampler
import time
from utils import log_on_main, get_git_info
from modeling import get_model_tokenizer, REGISTERED_MODEL_CLASSES
from dataset import REGISTERED_DATASET_CLASSES, REGISTERED_DATASET_COLLATORS

class NoShuffleTrainer(Trainer):
    def _get_train_sampler(self) -> Optional[Sampler]:
        return None
    
def train_model(args):

    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", -1))

    WORLD_RANK = int(os.environ.get("WORLD_RANK", -1))

    print(WORLD_RANK, 1)

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    print(WORLD_RANK, 2)
    
    training_type = config["training_type"]
    learning_rate = config["learning_rate"]
    # Microbatch size
    batch_size = config["batch_size"]
    # HF data path
    train_data_path = config["train_data_path"]
    dataset_type = config["dataset_type"]
    data_collator_type = config["data_collator_type"]
    base_model_name = config["base_model_name"]
    # Prompts will be truncted to this length
    max_length = config["max_length"]
    gradient_accumulation_steps = config["gradient_accumulation_steps"]
    deepspeed_config_path = config["deepspeed_config_path"]

    # Optional
    output_dir = config.get("output_dir", "training_outputs")
    epochs = config.get("num_train_epochs", 1)
    lr_scheduler = config.get("lr_schedule", "constant")
    chat_template = config.get("chat_template", None)
    # If the tokenizer/model does not already have a pad token, this will be used.
    pad_token_if_none = config.get("pad_token_if_none", "<|pad|>")
    cls_token_if_none = config.get("cls_token_if_none", "<|cls|>")
    proj_name = config.get("proj_name", None)
    init_type = config.get("init_type", "reset_params")
    loss_type = config.get("loss_type", None)
    # Type of transformer, see model.py for options.
    transformer_type = config.get("transformer_type", None)
    model_type = config.get("model_type", None)
    new_special_tokens = config.get("new_special_tokens", {})
    shuffle_dataset = config.get("shuffle_dataset", False)
    seed = config.get("seed", 42)

    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(WORLD_RANK, 3)

    if not proj_name:
        proj_name = f"{base_model_name.split('/')[1]}_ts{int(time.time())}"
    
    output_path = os.path.join(output_dir, proj_name)

    if args.checkpoint:
        resume_from_checkpoint = args.checkpoint
        log_on_main("resuming from checkpoint")
    else:
        resume_from_checkpoint = False

    if not resume_from_checkpoint:
        version = 1
        while os.path.exists(output_path):
            output_path = output_path.replace(f"_{version - 1}", "")
            output_path = output_path + f"_{version}"
            version += 1
    else:
        output_path = os.path.dirname(resume_from_checkpoint)
    # danger, might need distributed barrier here

    print(WORLD_RANK, 4)

    with open(deepspeed_config_path) as fin:
        deepspeed_config = json.load(fin)

    random.seed(42)
    set_seed(42)

    training_args = TrainingArguments(
        output_dir=output_path,
        report_to="wandb",
        run_name=proj_name,
        num_train_epochs=epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_strategy="no" if args.save_steps == -1 else "steps",
        save_steps=None if args.save_steps == -1 else args.save_steps,
        save_only_model=args.save_only_model,
        eval_strategy="no",
        logging_strategy="steps",
        logging_steps=1,
        ddp_timeout=9999999,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        lr_scheduler_type=lr_scheduler,
        logging_dir="./logs",
        fp16=False,
        bf16=True,
        learning_rate=learning_rate,
        load_best_model_at_end=False,
        gradient_checkpointing=True,
        do_train=True,
        bf16_full_eval=True,
        save_safetensors=True,
        disable_tqdm=False,
        remove_unused_columns=False,
        deepspeed=deepspeed_config,
        seed=42,
        data_seed=42,
        local_rank=LOCAL_RANK,
        save_on_each_node=args.local_fs
    )

    print(WORLD_RANK, 5)

    tokenizer = get_model_tokenizer(
        base_model_name=base_model_name,
        pad_token_if_none=pad_token_if_none,
        chat_template=chat_template,
        new_special_tokens=new_special_tokens,
        cls_token_if_none=cls_token_if_none,
    )

    print(WORLD_RANK, 6)

    data_collator = REGISTERED_DATASET_COLLATORS[data_collator_type](tokenizer=tokenizer, max_length=max_length)

    dataset_cls = REGISTERED_DATASET_CLASSES[dataset_type]

    print(WORLD_RANK, 7)

    with training_args.main_process_first(local=args.local_fs):

        train_data = dataset_cls.get_dataset(train_data_path, shuffle=shuffle_dataset, seed=seed)

    print(WORLD_RANK, 8)

    if WORLD_RANK <= 0:
        # Document the configuration in the output path.
        os.makedirs(output_path, exist_ok=True)

        training_details = {
            'git': get_git_info(),
            'config': config,
            'training_args': training_args.to_dict()
        }

        with open(os.path.join(output_path, "training_config.json"), "w") as fout:
            json.dump(training_details, fout, indent=1)

    print(WORLD_RANK, 9)

    match training_type:

        case "sft":

            model_cls = AutoModelForCausalLM
        
        case "reward":

            model_cls = REGISTERED_MODEL_CLASSES[model_type](
                model_type=transformer_type,
                init_type=init_type,
                tokenizer=tokenizer,
            )

    print(WORLD_RANK, 10)
    if resume_from_checkpoint:

        log_on_main(f"Loading model from checkpoint: {resume_from_checkpoint}")

        model = model_cls.from_pretrained(
            resume_from_checkpoint,
        )

    else:

        model = model_cls.from_pretrained(
            base_model_name,
        )

    print(WORLD_RANK, 11)
    
    trainer = NoShuffleTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data.with_format("torch"),
        data_collator=data_collator,
        compute_loss_func=REGISTERED_LOSSES[loss_type] if loss_type != None else None
    )

    print(WORLD_RANK, 12)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Argument Parser")

    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--checkpoint", "-cp", type=str, default=None)
    parser.add_argument("--save-steps", "-ss", type=int, default=-1)
    parser.add_argument("--save-only-model", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--local-fs", "-lfs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local_rank", type=int, required=False)
    parser.add_argument("--local-rank", type=int, required=False)

    args = parser.parse_args()

    train_model(args)