# A simplified script to log non-smooth statistics.
import logging
import random
from itertools import chain
from pathlib import Path
import wandb
import datasets
import torch
import torch.nn.functional as F
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from loadit import LoadIt
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
import hydra
from omegaconf import OmegaConf, DictConfig
from argparse import Namespace
from typing import NamedTuple
import _utils
from optimizers import O2NCGD, O2NCAdamW

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.34.0.dev0")

logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def init_tokenizer(config: DictConfig):
    """Initializes tokenizer. config=config.model"""
    if config.tokenizer_name:
        tokenizer_name = config.tokenizer_name
    else:
        tokenizer_name = config.model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, use_fast=(not config.use_slow_tokenizer), trust_remote_code=config.trust_remote_code
    )
    return tokenizer


def init_dataloaders(tokenizer, accelerator, config: DictConfig):
    """Initializes datasets. config=config.dataset"""
    # Load raw datasets.
    data_files = []
    for i in range(10):
        data_files.append(f"{config.data_dir}/train/0"+str(i)+".jsonl")
    for i in range(10,30):
        data_files.append(f"{config.data_dir}/train/"+str(i)+".jsonl")
    raw_datasets_train = load_dataset(
        'json', data_files=data_files, cache_dir=config.cache_dir, streaming=config.streaming)
    raw_datasets_val = load_dataset(
        'json', data_files=[f"{config.data_dir}/val.jsonl"], cache_dir=config.cache_dir, streaming=config.streaming)

    # Tokenize raw datasets.
    def tokenize_function(examples):
        return tokenizer(examples['text'])

    with accelerator.main_process_first():
        tokenized_datasets_train= raw_datasets_train['train'].map(
            tokenize_function,
            batched=True,
            remove_columns=['text', 'meta'],
        )
        tokenized_datasets_val = raw_datasets_val['train'].map(
            tokenize_function,
            batched=True,
            remove_columns=['text', 'meta'],
        )

    # Check block size.
    block_size = tokenizer.model_max_length
    if block_size > 1024:
        logger.warning(
            "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
            " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
            " override this default with `--block_size xxx`."
        )
        block_size = 1024

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    with accelerator.main_process_first():
        lm_datasets_train = tokenized_datasets_train.map(
            group_texts,
            batched=True,
        )
        lm_datasets_val = tokenized_datasets_val.map(
            group_texts,
            batched=True,
        )

    # Construct dataloader.
    train_dataset = lm_datasets_train.with_format("torch")
    eval_dataset = lm_datasets_val.with_format("torch")

    if config.use_loadit:
        train_dataloader = LoadIt(None, root_dir=config.loadit_dir, max_workers=1)
    else:
        train_dataloader = DataLoader(
            train_dataset, collate_fn=default_data_collator, batch_size=config.batch_size_train
        )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=config.batch_size_eval
    )

    return train_dataloader, eval_dataloader


def init_gpt2(tokenizer, config: DictConfig):
    """Initializes GPT2 model. config=config.model"""
    model_conf = AutoConfig.from_pretrained(
        config.model_name,
        trust_remote_code=config.trust_remote_code
    )
    ## turn off dropout
    model_conf.attn_pdrop = 0.0
    model_conf.resid_pdrop = 0.0
    model_conf.embd_pdrop = 0.0
    model = AutoModelForCausalLM.from_config(model_conf)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    # wandb.watch(model)
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    return model


def init_bert(tokenizer, config: DictConfig):
    """Initializes Bert model. config=config.model"""
    model_conf = AutoConfig.from_pretrained(
        config.model_name,
        trust_remote_code=config.trust_remote_code
    )
    ## turn off dropout
    model_conf.dropout = 0.0
    model_conf.attention_probs_dropout_prob = 0.0
    model = AutoModelForMaskedLM.from_config(model_conf)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    # wandb.watch(model)
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    return model


def init_model(tokenizer, config: DictConfig):
    """Initializes model. config=config.model"""
    if config.model_name == "gpt2":
        return init_gpt2(tokenizer, config)
    if config.model_name == "bert-base-uncased":
        return init_bert(tokenizer, config)


def init_optimzier(model, config: DictConfig):
    """Initialize optimizer."""
    train_config = config.train
    config = config.optimizer
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    if config.name == 'adam':
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, 
            lr=config.lr
        )
    elif config.name == 'sgd':
        optimizer = torch.optim.SGD(
            optimizer_grouped_parameters, 
            lr=config.lr, 
            weight_decay=config.weight_decay, 
            momentum=config.beta
        )
    elif config.name == 'o2nc_ogd':
        optimizer = O2NCGD(
            optimizer_grouped_parameters, 
            lr=config.lr, 
            beta=config.beta, 
            mu=config.weight_decay,
            random_scaling=config.random_scaling
        )
    elif config.name == 'o2nc_adamw':
        optimizer = O2NCAdamW(
            optimizer_grouped_parameters, 
            lr=config.lr, 
            b1=0.9, b2=0.999, wd=config.weight_decay, eps=1e-8, 
            random_scaling=config.random_scaling
        )

    lr_scheduler = get_scheduler(
        name=config.schedule,
        optimizer=optimizer,
        num_warmup_steps=config.warmup * train_config.gradient_accumulation_steps,
        num_training_steps=train_config.max_steps * train_config.gradient_accumulation_steps
    )
    return optimizer, lr_scheduler


class Logstate(NamedTuple):
    iteration: int
    params_diff: dict
    delta: dict
    random_scalar: torch.Tensor
    logs: dict


def init_logstate(model):
    logstate = Logstate(
        iteration=1,
        params_diff={name: torch.zeros_like(param) for name, param in model.named_parameters()},
        delta={name: torch.zeros_like(param) for name, param in model.named_parameters()},
        random_scalar=torch.ones([]),
        logs = {
            "f(x_t,z_t)": 0.0,
            "f(x_t,z_t)_avg": 0.0,
            "smooth/<g_t, x_t-x_{t-1}>": 0.0,
            "smooth/<g_t, x_t-x_{t-1}>_sum": 0.0,
            "smooth/<g_t, Delta_t>": 0.0,
            "smooth/<g_t, Delta_t>_sum": 0.0,
            "smooth/f(x_t,z_t)-f(x_{t-1},z_t)": 0.0,
            "smooth/f(x_t,z_t)-f(x_{t-1},z_t)_sum": 0.0,
            "norm/s_t": 1.0,
            "norm/|x_t-x_{t-1}|": 0.0,
            "norm/|Delta_t|": 0.0,
            "norm/|g_t|": 0.0,
            "sancheck/|Delta_t|": 0.0,
            "sancheck/<g_t, Delta_t>": 0.0,
        }
    )
    return logstate


def loss_fn(model, batch, use_hugging_face=True):
    """Wrapper of loss function: if not using hugging face model (e.g., mingpt), then manually cmopute loss."""
    if use_hugging_face:
        return model(**batch).loss
    else:
        idx, targets = batch["input_ids"], batch["labels"]
        logits, _ = model(idx)
        # SHIFT the labels by 1 and drop the last label.
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = targets[..., 1:].contiguous()
        return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


def train_step(
    logstate,
    model,
    batch,
    optimizer,
    lr_scheduler,
    accelerator,
    config
) -> Logstate:
    clip_norm = config.train.gradient_clip_norm
    use_hugging_face = config.model.use_hugging_face

    model.train()
    with accelerator.accumulate(model):
        # ==========================================================================
        # Auxilliary computation for logging.
        params_diff = logstate.params_diff                                                      # x_t-x_{t-1}
        current_params = {name: param.data.clone() for name, param in model.named_parameters()} # x_t
        prev_params = _utils.tree_subtract(current_params, params_diff)                          # x_{t-1}
        # Compute f(x_{t-1},z_t)
        optimizer.zero_grad()
        try:
            # Replace model parameters with cloned parameters
            for name, param in model.named_parameters():
                param.data.copy_(prev_params[name])
            with torch.no_grad():
                # prev_loss = model(**batch).loss.detach().float()
                prev_loss = loss_fn(model, batch, use_hugging_face)
        finally:
            # Restore original parameters
            for name, param in model.named_parameters():
                param.data.copy_(current_params[name])

        # ==========================================================================
        # Actual training.
        optimizer.zero_grad()
        # loss = model(**batch).loss
        loss = loss_fn(model, batch, use_hugging_face)
        accelerator.backward(loss)
        if clip_norm:               # optional gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        lr_scheduler.step()         # update x_t to x_{t+1}, computes s_{t+1} and Delta_{t+1}

        current_loss = loss.detach().float()                                                    # f(x_t,z_t)
        grads = {name: param.grad.clone() for name, param in model.named_parameters()}          # g(x_t, z_t)
        new_scalar = optimizer.optimizer.scalar                                                 # s_{t+1}
        new_delta = _utils.get_opt_state("Delta", optimizer.optimizer, model)                    # Delta_{t+1}=(x_{t+1}-x_t)/s_{t+1}
        next_params = {name: param.data.clone() for name, param in model.named_parameters()}    # x_{t+1}
        new_params_diff = _utils.tree_subtract(next_params, current_params)                      # x_{t+1}-x_t

        # ==========================================================================
        # Compute logging statistics.
        logs = logstate.logs
        iteration = logstate.iteration                                                          # t
        random_scalar = logstate.random_scalar                                                  # s_t
        params_diff = logstate.params_diff                                                      # x_t-x_{t-1}
        delta = logstate.delta                                                                  # Delta_t

        avg_loss = (logs["f(x_t,z_t)_avg"] * (iteration-1) + current_loss) / iteration
        inner_g_dx = _utils.tree_inner(grads, params_diff)                                       # <g_t, x_t-x_{t-1}>
        inner_g_dx_sum = logs["smooth/<g_t, x_t-x_{t-1}>_sum"] + inner_g_dx
        inner_g_delta = _utils.tree_inner(grads, delta)                                          # <g_t, Delta_t>
        inner_g_delta_sum = logs["smooth/<g_t, Delta_t>_sum"] + inner_g_delta
        loss_diff = current_loss - prev_loss                                                    # f(x_t,z_t)-f(x_{t-1},z_t)
        loss_diff_sum = logs["smooth/f(x_t,z_t)-f(x_{t-1},z_t)_sum"] + loss_diff
        norm_dx = _utils.tree_norm_l2(params_diff)
        norm_delta = _utils.tree_norm_l2(delta)
        logs.update({
            "f(x_t,z_t)": current_loss,
            "f(x_t,z_t)_avg": avg_loss,
            "smooth/<g_t, x_t-x_{t-1}>": inner_g_dx,
            "smooth/<g_t, x_t-x_{t-1}>_sum": inner_g_dx_sum,
            "smooth/<g_t, Delta_t>": inner_g_delta,
            "smooth/<g_t, Delta_t>_sum": inner_g_delta_sum,
            "smooth/f(x_t,z_t)-f(x_{t-1},z_t)": loss_diff,
            "smooth/f(x_t,z_t)-f(x_{t-1},z_t)_sum": loss_diff_sum,
            "norm/s_t": random_scalar,
            "norm/|x_t-x_{t-1}|": norm_dx,
            "norm/|Delta_t|": norm_delta,
            "norm/|g_t|": _utils.tree_norm_l2(grads),
            "sancheck/|Delta_t|": norm_dx - random_scalar*norm_delta,
            "sancheck/<g_t, Delta_t>": inner_g_dx - random_scalar*inner_g_delta,
        })
    
    return Logstate(
        iteration=iteration+1,
        params_diff=new_params_diff,
        delta=new_delta,
        random_scalar=new_scalar,
        logs=logs,
    )


def train(config: DictConfig) -> None:
    send_example_telemetry("run_clm_no_trainer", Namespace(**config))

    # Initialize pytorch accelerator
    accelerator = Accelerator(gradient_accumulation_steps=config.train.gradient_accumulation_steps)
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Set training seed.
    if config.train.seed:
        set_seed(config.train.seed)

    accelerator.wait_for_everyone()

    # ===============================================================================================
    # Training starts here...
    tokenizer = init_tokenizer(config.model)
    model = init_model(tokenizer, config.model)
    train_dataloader = LoadIt(root_dir=config.dataset.loadit_dir, max_workers=1)
    optimizer, lr_scheduler = init_optimzier(model, config)
    
    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()
    
    # Total number of data points
    num_data_points = 1000000

    # Number of indices to sample
    sample_size = config.train.max_steps
    sampled_indices = random.sample(range(num_data_points), sample_size)

    # Main train loop.
    logstate = init_logstate(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pbar = tqdm(range(config.train.max_steps))
    for _ in range(config.train.epochs):
        for i in range(len(sampled_indices)):
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in train_dataloader[sampled_indices[i]].items()}
            logstate = train_step(logstate, model, batch, optimizer, lr_scheduler, accelerator, config)

            pbar.set_description(f"iteration: {logstate.iteration-1}, avg_train_loss: {logstate.logs['f(x_t,z_t)_avg']}")
            if config.logging.wandb_project:
                wandb.log(logstate.logs, step=logstate.iteration)
            if logstate.iteration > config.train.max_steps:
                break


@hydra.main(version_base=None, config_path="", config_name="config")
def main(config: DictConfig) -> None:
    logging.info(OmegaConf.to_yaml(config))
    if config.logging.wandb_project:
        wandb.init(project=config.logging.wandb_project, name=config.logging.wandb_name)
        wandb.config.update(OmegaConf.to_container(config))
    train(config)


if __name__ == "__main__":
    main()
