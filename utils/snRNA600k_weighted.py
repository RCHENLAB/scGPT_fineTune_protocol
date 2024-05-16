# %%
import copy
import gc
import json
import os
import shutil
from pathlib import Path
import time
from typing import List, Tuple, Dict, Union, Optional
import warnings
from multiprocessing import Pool
import torch
import scanpy as sc
import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)

import scgpt as scg
from scgpt.model import TransformerModel, AdversarialDiscriminator
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import set_seed, category_str2int, eval_scib_metrics

from data_collator import *

sc.set_figure_params(figsize=(6, 6))
warnings.filterwarnings('ignore')
os.environ["KMP_WARNINGS"] = "off"
os.environ["WANDB_MODE"] = "offline"


# %%
hyperparameter_defaults = dict(
    seed=0,
    dataset_name="snRNA600k",
    do_train=True,
    load_model="../scgpt_human",
    mask_ratio=0.0,
    epochs=15,
    n_bins=51,
    MVC=False,  # Masked value prediction for cell embedding
    ecs_thres=0.0,  # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
    dab_weight=0.0,
    lr=5e-5,
    batch_size=64,
    layer_size=128,
    nlayers=4,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead=4,  # number of heads in nn.MultiheadAttention
    dropout=0.2,  # dropout probability
    schedule_ratio=0.96,  # ratio of epochs for learning rate schedule
    save_eval_interval=3,
    fast_transformer=False,  # or True if flash-attn is installed
    pre_norm=False,
    amp=True,  # Automatic Mixed Precision
    include_zero_gene=False,
    freeze=False,  # freeze
    DSBN=False,  # Domain-spec batchnorm
)

# %%
run = wandb.init(
    config=hyperparameter_defaults,
    project="scgpt-finetune",
    reinit=True,
    settings=wandb.Settings(start_method="fork"),
    # name='normed_batch64_weighted'
)
config = wandb.config
print(config)

set_seed(config.seed)

# %%

# settings for input and preprocessing
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
mask_ratio = config.mask_ratio
mask_value = "auto"  # for masked values, now it should always be auto

include_zero_gene = config.include_zero_gene  # if True, include zero genes among hvgs in the training
max_seq_len = 1201
n_bins = config.n_bins

# input/output representation
input_style = "binned"  # "normed_raw", "log1p", or "binned"
output_style = "binned"  # "normed_raw", "log1p", or "binned"

# settings for training
MLM = False  # whether to use masked language modeling, currently it is always on.
CLS = True  # celltype classification objective
ADV = False  # Adversarial training for batch correction
CCE = False  # Contrastive cell embedding objective
MVC = config.MVC  # Masked value prediction for cell embedding
ECS = config.ecs_thres > 0  # Elastic cell similarity objective
DAB = False  # Domain adaptation by reverse backpropagation, set to 2 for separate optimizer
INPUT_BATCH_LABELS = False  # TODO: have these help MLM and MVC, while not to classifier
input_emb_style = "continuous"  # "category" or "continuous" or "scaling"
cell_emb_style = "cls"  # "avg-pool" or "w-pool" or "cls"
adv_E_delay_epochs = 0  # delay adversarial training on encoder for a few epochs
adv_D_delay_epochs = 0
mvc_decoder_style = "inner product"
ecs_threshold = config.ecs_thres
dab_weight = config.dab_weight

explicit_zero_prob = MLM and include_zero_gene  # whether explicit bernoulli for zeros
do_sample_in_train = False and explicit_zero_prob  # sample the bernoulli in training
do_sample_in_collator = True

per_seq_batch_sample = False

# settings for optimizer
lr = config.lr  # TODO: test learning rate ratio between two tasks
lr_ADV = 1e-3  # learning rate for discriminator, used when ADV is True
batch_size = config.batch_size
eval_batch_size = config.batch_size
epochs = config.epochs
schedule_interval = 1

# settings for the model
fast_transformer = config.fast_transformer
fast_transformer_backend = "flash"  # "linear" or "flash"
embsize = config.layer_size  # embedding dimension
d_hid = config.layer_size  # dimension of the feedforward network in TransformerEncoder
nlayers = config.nlayers  # number of TransformerEncoderLayer in TransformerEncoder
nhead = config.nhead  # number of heads in nn.MultiheadAttention
dropout = config.dropout  # dropout probability

# logging
log_interval = 100  # iterations
save_eval_interval = config.save_eval_interval  # epochs
do_eval_scib_metrics = True

# %%
assert input_style in ["normed_raw", "log1p", "binned"]
assert output_style in ["normed_raw", "log1p", "binned"]
assert input_emb_style in ["category", "continuous", "scaling"]
if input_style == "binned":
    if input_emb_style == "scaling":
        raise ValueError("input_emb_style `scaling` is not supported for binned input.")
elif input_style == "log1p" or input_style == "normed_raw":
    if input_emb_style == "category":
        raise ValueError(
            "input_emb_style `category` is not supported for log1p or normed_raw input."
        )

if input_emb_style == "category":
    mask_value = n_bins + 1
    pad_value = n_bins  # for padding gene expr values
    n_input_bins = n_bins + 2
else:
    mask_value = -1
    pad_value = -2
    n_input_bins = n_bins

if ADV and DAB:
    raise ValueError("ADV and DAB cannot be both True.")
DAB_separate_optim = True if DAB > 1 else False

# %%
dataset_name = config.dataset_name
save_dir = Path(f"./save/dev_{dataset_name}-{time.strftime('%b%d-%H-%M')}/")
save_dir.mkdir(parents=True, exist_ok=True)
print(f"save to {save_dir}")
# save the whole script to the dir
os.system(f"cp {__file__} {save_dir}")

logger = scg.logger
scg.utils.add_file_handler(logger, save_dir / "run.log")

# %%
# ## Loading and preparing data
data_dir = Path("../data")
file_name = "TRAIN_snRNA600K.h5ad"
file_dir = data_dir / file_name
do_norm = False
logger.info(f'Start reading data from {file_dir}...')
adata = sc.read_h5ad(file_dir, backed='r')  # data streaming in backed mode saves ~10% memory usage
filter_gene_by_counts = False

# make the batch category column
# adata.obs["celltype"] = adata.obs["majorclass"].astype("category")
adata.var["gene_name"] = adata.var.index.tolist()
celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
celltypes = adata.obs["celltype"].unique()
num_types = len(np.unique(celltype_id_labels))
id2type = dict(enumerate(adata.obs["celltype"].astype("category").cat.categories))
adata.obs["celltype_id"] = celltype_id_labels
num_batch_types = 1


# %%
logger.info(f'Start loading pre-trained model and weights ...')
if config.load_model is not None:
    model_dir = Path(config.load_model)
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    vocab_file = model_dir / "vocab.json"
    shutil.copy(vocab_file, save_dir / "vocab.json")
    shutil.copy(model_config_file, save_dir / "args.json")
    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in adata.var["gene_name"]
    ]
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    logger.info(
        f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}."
    )
    adata = adata[:, adata.var["id_in_vocab"] >= 0]

    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    logger.info(
        f"Resume model from {model_file}, the model args will be override by the "
        f"config {model_config_file}."
    )
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]
else:
    embsize = config.layer_size
    nhead = config.nhead
    nlayers = config.nlayers
    d_hid = config.layer_size


# %%
genes = np.array(vocab(adata.var["gene_name"].tolist()), dtype=int)
cell_type_ids = adata.obs['celltype_id']
if do_norm:
    logger.info(f'Starting data normalization ...')
    start_norm = time.time()
    X = sc.pp.normalize_total(adata.to_memory(), target_sum=1e4, inplace=False)['X']
    del adata
    sc.pp.log1p(X)
    end_norm = time.time()
    logger.info(f'Normalization took {end_norm - start_norm:.2f} seconds')
else:
    logger.info(f'Skipping data normalization ...')
    X = adata.X[:, :]
    if adata:
        del adata


# %%
def prepare_data(X, cell_type_ids):
    data_samples = []
    num_processes = 4  # mp.cpu_count()
    logger.info(f'Loading raw counts into chunks for batch processing ...')
    start_dense_x = time.time()
    dense_X = X.A
    end_dense_x = time.time()
    logger.info(f'Transfer X into memory by {end_dense_x - start_dense_x:.2f} seconds ... ')
    chunk_size = 512
    total_samples = X.shape[0]
    num_batches = (total_samples + chunk_size - 1) // chunk_size
    chunks = [range(i * chunk_size, min((i + 1) * chunk_size, total_samples)) for i in range(num_batches)]
    parallel_chunks = []
    with tqdm(total=len(chunks), desc="Appending chunks") as progress_bar:
        for chunk in chunks:
            parallel_chunks.append((dense_X[chunk], cell_type_ids[chunk]))
            progress_bar.update(1)
    logger.info(
        f'Creating multiprocessing pool ({num_processes} process) ...'
    )
    with Pool(processes=num_processes) as pool:
        partial_datasets = pool.starmap(process_chunk, parallel_chunks)
    logger.info(f'Collecting results ...')
    for partial_dataset in partial_datasets:
        data_samples.extend(partial_dataset)

    # free memory
    del chunk_size
    del chunks
    del total_samples
    del num_batches
    del parallel_chunks
    del partial_datasets
    del dense_X
    del X
    del cell_type_ids

    return data_samples


# %%
logger.info(f'Preparing data for training ...')
raw_dataset = prepare_data(X, cell_type_ids)
vocab.set_default_index(vocab["<pad>"])
logger.info(f'Data is ready ...')

# %%
(
    train_data,
    valid_data
) = train_test_split(
    raw_dataset, test_size=0.1, shuffle=True
)


# %%
collator = DataCollator(
    do_padding=True if max_seq_len is not None else False,
    pad_token_id=vocab[pad_token],
    pad_value=pad_value,
    do_mlm=MLM,
    do_binning=True if input_style == "binned" else False,
    mask_value=mask_value,
    max_length=max_seq_len,
    sampling=do_sample_in_collator,
    keep_first_n_tokens=0,
    data_style='cls',
    filtered_gene_list=genes
)

train_sampler = (RandomSampler(train_data))

train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    sampler=train_sampler,
    collate_fn=collator,
    drop_last=False,
    num_workers=min(len(os.sched_getaffinity(0)), batch_size),
    pin_memory=True,
    prefetch_factor=4,
)

valid_sampler = (
    SequentialSampler(valid_data)
)

valid_loader = DataLoader(
    valid_data,
    batch_size=batch_size,
    sampler=valid_sampler,
    collate_fn=collator,
    drop_last=False,
    num_workers=min(len(os.sched_getaffinity(0)), eval_batch_size),
    pin_memory=True,
)


# %%
logger.info(f'Create new model instance ...')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_types = len(np.unique(celltype_id_labels))
ntokens = len(vocab)  # size of vocabulary
model = TransformerModel(
    ntokens,
    embsize,
    nhead,
    d_hid,
    nlayers,
    nlayers_cls=3,
    n_cls=num_types if CLS else 1,
    vocab=vocab,
    dropout=dropout,
    pad_token=pad_token,
    pad_value=pad_value,
    do_mvc=MVC,
    do_dab=DAB,
    use_batch_labels=INPUT_BATCH_LABELS,
    num_batch_labels=num_batch_types,
    domain_spec_batchnorm=config.DSBN,
    input_emb_style=input_emb_style,
    n_input_bins=n_input_bins,
    cell_emb_style=cell_emb_style,
    mvc_decoder_style=mvc_decoder_style,
    ecs_threshold=ecs_threshold,
    explicit_zero_prob=explicit_zero_prob,
    use_fast_transformer=fast_transformer,
    fast_transformer_backend=fast_transformer_backend,
    pre_norm=config.pre_norm,
)
if config.load_model is not None:
    try:
        model.load_state_dict(torch.load(model_file))
        logger.info(f"Loading all model params from {model_file}")
    except:
        # only load params that are in the model and match the size
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file)
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        for k, v in pretrained_dict.items():
            logger.info(f"Loading params {k} with shape {v.shape}")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

pre_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())

# Freeze all pre-decoder weights
for name, para in model.named_parameters():
    print("-" * 20)
    print(f"name: {name}")
    if config.freeze and "encoder" in name and "transformer_encoder" not in name:
        # if config.freeze and "encoder" in name:
        print(f"freezing weights for: {name}")
        para.requires_grad = False

post_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())

logger.info(f"Total Pre freeze Params {(pre_freeze_param_count)}")
logger.info(f"Total Post freeze Params {(post_freeze_param_count)}")
wandb.log(
    {
        "info/pre_freeze_param_count": pre_freeze_param_count,
        "info/post_freeze_param_count": post_freeze_param_count,
    },
)

model.to(device)
is_parallel = torch.cuda.device_count() > 1
if is_parallel:
    model = nn.DataParallel(model)

wandb.watch(model)

if ADV:
    discriminator = AdversarialDiscriminator(
        d_model=embsize,
        n_cls=num_batch_types,
    ).to(device)


# %%
logger.info(f'Define loss metrics and optimizer ...')
class_counts = np.bincount(celltype_id_labels)
imbalance_factor = np.sqrt(class_counts.max() / class_counts)
imbalance_factor *= 10
class_weights = 1 / (class_counts * imbalance_factor)
class_weights /= class_weights.sum()
criterion = masked_mse_loss
criterion_cls = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float, device=device))
criterion_dab = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float, device=device))
optimizer = torch.optim.Adam(
    model.parameters(), lr=lr, eps=1e-4 if config.amp else 1e-8
)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, schedule_interval, gamma=config.schedule_ratio
)

scaler = torch.cuda.amp.GradScaler(enabled=config.amp)


# %%
def train(model: nn.Module, loader: DataLoader) -> None:
    """
    Train the model for one epoch.
    """
    model.train()
    (
        total_loss,
        total_mse,
        total_cls,
        total_cce,
        total_mvc,
        total_ecs,
        total_dab,
        total_adv_E,
        total_adv_D,
        total_zero_log_prob,
        total_mvc_zero_log_prob,
    ) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    total_error = 0.0
    start_time = time.time()

    num_batches = len(loader)
    for batch, batch_data in enumerate(loader):
        global_iter = epoch * num_batches + batch
        batch_data = {k: v.to(device) for k, v in batch_data.items()}
        input_gene_ids = batch_data["input_gene_ids"].to(device)
        input_values = batch_data["input_expr"].to(device)
        target_values = batch_data["target_expr"].to(device)
        batch_labels = None
        celltype_labels = batch_data["cell_types"].to(device)

        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
        with torch.cuda.amp.autocast(enabled=config.amp):
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels if INPUT_BATCH_LABELS or config.DSBN else None,
                CLS=CLS,
                CCE=CCE,
                MVC=MVC,
                ECS=ECS,
                do_sample=do_sample_in_train
            )

            masked_positions = input_values.eq(mask_value)  # the postions to predict
            loss = 0.0
            metrics_to_log = {}
            if MLM:
                loss_mse = criterion(
                    output_dict["mlm_output"], target_values, masked_positions
                )
                metric_loss_mse = torch.mean(torch.sum(loss_mse)) if is_parallel else loss_mse
                loss = loss + metric_loss_mse
                metrics_to_log = {"train/mse": metric_loss_mse.item()}
            if explicit_zero_prob:
                loss_zero_log_prob = criterion_neg_log_bernoulli(
                    torch.nan_to_num(output_dict["mlm_zero_probs"]), target_values, masked_positions
                )
                loss = loss + loss_zero_log_prob
                metrics_to_log.update({"train/nzlp": loss_zero_log_prob.item()})
            if CLS:
                loss_cls = criterion_cls(output_dict["cls_output"], celltype_labels)
                metric_loss_cls = torch.mean(torch.sum(loss_cls)) if is_parallel else loss_cls
                loss = loss + metric_loss_cls
                metrics_to_log.update({"train/cls": metric_loss_cls.item()})

                error_rate = 1 - (
                    (output_dict["cls_output"].argmax(1) == celltype_labels)
                    .sum()
                    .item()
                ) / celltype_labels.size(0)
            if CCE:
                loss_cce = 10 * output_dict["loss_cce"]
                metric_loss_cce = torch.mean(torch.sum(loss_cce)) if is_parallel else loss_cce
                loss = loss + metric_loss_cce
                metrics_to_log.update({"train/cce": metric_loss_cce.item()})
            if MVC:
                loss_mvc = criterion(
                    output_dict["mvc_output"], target_values, masked_positions
                )
                metric_loss_mvc = torch.mean(torch.sum(loss_mvc)) if is_parallel else loss_mvc
                loss = loss + metric_loss_mvc
                metrics_to_log.update({"train/mvc": metric_loss_mvc.item()})
            if MVC and explicit_zero_prob:
                loss_mvc_zero_log_prob = criterion_neg_log_bernoulli(
                    output_dict["mvc_zero_probs"], target_values, masked_positions
                )
                metric_loss_mvc_zero_log_prob = torch.mean(torch.sum(loss_mvc_zero_log_prob)) if is_parallel else loss_mvc_zero_log_prob
                loss = loss + metric_loss_mvc_zero_log_prob
                metrics_to_log.update({"train/mvc_nzlp": metric_loss_mvc_zero_log_prob.item()})
            if ECS:
                loss_ecs = 10 * output_dict["loss_ecs"]
                metric_loss_ecs = torch.mean(torch.sum(loss_ecs)) if is_parallel else loss_ecs
                loss = loss + metric_loss_ecs
                metrics_to_log.update({"train/ecs": metric_loss_ecs.item()})
            if DAB:
                # try weighting and separate optimizer
                loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)
                metric_loss_dab = torch.mean(torch.sum(loss_dab)) if is_parallel else loss_dab
                loss = loss + dab_weight * metric_loss_dab
                metrics_to_log.update({"train/dab": metric_loss_dab.item()})

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
                error_if_nonfinite=False if scaler.is_enabled() else True,
            )
            if len(w) > 0:
                logger.warning(
                    f"Found infinite gradient. This may be caused by the gradient "
                    f"scaler. The current scale is {scaler.get_scale()}. This warning "
                    "can be ignored if no longer occurs after autoscaling of the scaler."
                )
        scaler.step(optimizer)
        scaler.update()

        wandb.log(metrics_to_log)

        total_loss += loss.item()
        total_mse += loss_mse.item() if MLM else 0.0
        total_cls += loss_cls.item() if CLS else 0.0
        total_cce += loss_cce.item() if CCE else 0.0
        total_mvc += loss_mvc.item() if MVC else 0.0
        total_ecs += loss_ecs.item() if ECS else 0.0
        total_dab += loss_dab.item() if DAB else 0.0
        total_zero_log_prob += loss_zero_log_prob.item() if explicit_zero_prob else 0.0
        total_mvc_zero_log_prob += (
            loss_mvc_zero_log_prob.item() if MVC and explicit_zero_prob else 0.0
        )
        total_error += error_rate
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval
            cur_cls = total_cls / log_interval if CLS else 0.0
            cur_cce = total_cce / log_interval if CCE else 0.0
            cur_mvc = total_mvc / log_interval if MVC else 0.0
            cur_ecs = total_ecs / log_interval if ECS else 0.0
            cur_dab = total_dab / log_interval if DAB else 0.0
            cur_adv_E = total_adv_E / log_interval if ADV else 0.0
            cur_adv_D = total_adv_D / log_interval if ADV else 0.0
            cur_zero_log_prob = (
                total_zero_log_prob / log_interval if explicit_zero_prob else 0.0
            )
            cur_mvc_zero_log_prob = (
                total_mvc_zero_log_prob / log_interval
                if MVC and explicit_zero_prob
                else 0.0
            )
            cur_error = total_error / log_interval
            # ppl = math.exp(cur_loss)
            logger.info(
                f"|\tepoch\t{epoch:3d}\t|\t{batch:3d}/{num_batches:3d} batches\t|\t"
                f"lr\t{lr:.4e}\t|\tms/batch\t{ms_per_batch:5.2f}\t|\t"
                f"loss\t{cur_loss:5.2f}\t|\t"
                + (f"mse\t{cur_mse:5.2f}\t|\tmre\t{cur_error:5.2f}\t|\t" if MLM else "")
                + (f"cls\t{cur_cls:5.2f}\t|\t" if CLS else "")
                + (f"err\t{cur_error:5.2f}\t|\t" if CLS else "")
                + (f"cce\t{cur_cce:5.2f}\t|\t" if CCE else "")
                + (f"mvc\t{cur_mvc:5.2f}\t|\t" if MVC else "")
                + (f"ecs\t{cur_ecs:5.2f}\t|\t" if ECS else "")
                + (f"dab\t{cur_dab:5.2f}\t|\t" if DAB else "")
                + (f"adv_E\t{cur_adv_E:5.2f}\t|\t" if ADV else "")
                + (f"adv_D\t{cur_adv_D:5.2f}\t|\t" if ADV else "")
                + (f"nzlp\t{cur_zero_log_prob:5.2f}\t|\t" if explicit_zero_prob else "")
                + (f"mvc_nzlp\t{cur_mvc_zero_log_prob:5.2f}\t|\t" if MVC and explicit_zero_prob else "")
            )
            total_loss = 0
            total_mse = 0
            total_cls = 0
            total_cce = 0
            total_mvc = 0
            total_ecs = 0
            total_dab = 0
            total_adv_E = 0
            total_adv_D = 0
            total_zero_log_prob = 0
            total_mvc_zero_log_prob = 0
            total_error = 0
            start_time = time.time()


def define_wandb_metrcis():
    wandb.define_metric("valid/mse", summary="min", step_metric="epoch")
    wandb.define_metric("valid/mre", summary="min", step_metric="epoch")
    wandb.define_metric("valid/dab", summary="min", step_metric="epoch")
    wandb.define_metric("valid/sum_mse_dab", summary="min", step_metric="epoch")
    wandb.define_metric("test/avg_bio", summary="max")


def evaluate(model: nn.Module, loader: DataLoader, return_raw: bool = False) -> float:
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    total_loss = 0.0
    total_error = 0.0
    total_dab = 0.0
    total_num = 0
    predictions = []
    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["input_gene_ids"].to(device)
            input_values = batch_data["input_expr"].to(device)
            target_values = batch_data["target_expr"].to(device)
            batch_labels = None  # batch_data["target_expr"].to(device)
            celltype_labels = batch_data["cell_types"].to(device)

            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            with torch.cuda.amp.autocast(enabled=config.amp):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if INPUT_BATCH_LABELS or config.DSBN else None,
                    CLS=CLS,  # evaluation does not need CLS or CCE
                    CCE=False,
                    MVC=False,
                    ECS=False,
                    do_sample=do_sample_in_train,
                    # generative_training = False,
                )
                output_values = output_dict["cls_output"]
                loss = criterion_cls(output_values, celltype_labels)

                if DAB:
                    loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)

            total_loss += loss.item() * len(input_gene_ids)
            accuracy = (output_values.argmax(1) == celltype_labels).sum().item()
            total_error += (1 - accuracy / len(input_gene_ids)) * len(input_gene_ids)
            total_dab += loss_dab.item() * len(input_gene_ids) if DAB else 0.0
            total_num += len(input_gene_ids)
            preds = output_values.argmax(1).cpu().numpy()
            predictions.append(preds)

    wandb.log(
        {
            "valid/mse": total_loss / total_num,
            "valid/err": total_error / total_num,
            "valid/dab": total_dab / total_num,
            "valid/sum_mse_dab": (total_loss + dab_weight * total_dab) / total_num,
            "epoch": epoch,
        },
    )

    if return_raw:
        return np.concatenate(predictions, axis=0)

    return total_loss / total_num, total_error / total_num


# %%
logger.info(f'Start training ...')
best_val_loss = float("inf")
best_avg_bio = 0.0
best_model = None
define_wandb_metrcis()

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()

    if config.do_train:
        train(
            model,
            loader=train_loader,
        )
    val_loss, val_err = evaluate(
        model,
        loader=valid_loader,
    )
    elapsed = time.time() - epoch_start_time
    logger.info("-" * 89)
    logger.info(
        f"|\tend of epoch:\t{epoch:3d}\t|\ttime:\t{elapsed:5.2f}s\t|\t"
        f"valid loss/mse:\t{val_loss:5.4f}\t|\terr:\t{val_err:5.4f}"
    )
    logger.info("-" * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)
        best_model_epoch = epoch
        logger.info(f"Best model with score {best_val_loss:5.4f}")

    if epoch % config.save_eval_interval == 0 or epoch == config.epochs:
        logger.info(f"Saving model to {save_dir}")
        torch.save(best_model.state_dict(), save_dir / f"model_e{best_model_epoch}.pt")

    scheduler.step()

# %%
# save the best model
torch.save(best_model.state_dict(), save_dir / "best_model.pt")
artifact = wandb.Artifact(f"best_model", type="model")
glob_str = os.path.join(save_dir, "best_model.pt")
artifact.add_file(glob_str)
run.log_artifact(artifact)
logger.info(f'Training job is saved to {save_dir}')

run.finish()
wandb.finish()
gc.collect()

# %%
# END
