# %% Dependencies Imports
import copy
import gc
import json
import pickle
import os
import sys
import shutil
import time
import warnings
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import yaml
import click
from anndata import AnnData
from numpy import ndarray, dtype
from pathlib import Path
from typing import *
from tqdm import tqdm
from dataclasses import dataclass, field

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset, DataLoader, BatchSampler, RandomSampler, SequentialSampler
from torchtext.vocab import Vocab
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, cohen_kappa_score
from scipy.sparse import issparse

import scgpt as scg
from scgpt import SubsetsBatchSampler
from scgpt.preprocess import binning
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.model import TransformerModel, AdversarialDiscriminator
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import set_seed, category_str2int, eval_scib_metrics
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli
)

sc.set_figure_params(figsize=(6, 6))
warnings.filterwarnings('ignore')
os.environ["KMP_WARNINGS"] = "off"