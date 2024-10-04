# Fine-tune Protocol for eye-scGPT
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13863911.svg)](https://doi.org/10.5281/zenodo.13863911)

`Maintainer of the protocol: Shanli Ding`

This is a protocol for doing fine-tuning on any single-cell dataset `(ex: .h5ad, .hd5f, etc.)` with [scGPT](https://www.nature.com/articles/s41592-024-02201-0).

## Quick How-to-Use Guide
* **Datasets**
  1. Select and download train/eval datasets from [HERE](https://zenodo.org/records/13863911)
* **Fine-tuning Custom scGPT Model**
  1. Prepare the dataset for fine-tuning task. Run `protocol_preprocess.py`
  2. Run `protocol_finetune.py`

* **Inference on Trained Model**
  1. Prepare the dataset for inference task. Run `protocol_preprocess.py`
  2. Run `protocol_inference.py`
 
* **Zero-shot Inference on scGPT**
  1. Download scGPT index file: https://drive.google.com/drive/folders/1q14U50SNg5LMjlZ9KH-n-YsGRi8zkCbe
  2. Run `protocol_zeroshot_inference.py`
  3. Or you can use the interactive Jupyter Notebook `./scGPT_fineTune_protcol/notebooks/protocol_notebook.ipynb`

## Main Scripts
1. **Pre-process** \
   Prepare the custom dataset to train-ready state for next fine-tuning step. \
   To see full help information on how to use preprocess.py script use this command:
   ```bash
    python protocol_preprocess.py --help
   ```
   Pre-process example command:
   ```bash
    python protocol_preprocess.py \
     --dataset_directory=../datasets/retina_snRNA.h5ad \
     --cell_type_col=celltype \
     --batch_id_col=sampleid \
     --load_model=../scGPT_human \
      --wandb_sync=True \
     --wandb_project=finetune_retina_snRNA \
     --wandb_name=finetune_example1
   ```
   > In **pre-process** step, *--load_model* has to be one of the [pre-trained scGPT models](https://github.com/bowang-lab/scGPT?tab=readme-ov-file#pretrained-scgpt-model-zoo). Please download the
   approriate pre-trained scGPT model, and it is recommended to use [*scGPT_Human*](https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y) model for any fine-tuning task.

2. **Fine-tune** \
   Start fine-tuning the foundation scGPT model with your custom dataset. Here we are introducing our eye-scGPT that is trained specific on
   the human retina single-nuclei and single-cell datasets. Please adjust any parameters with your own requirements. \
   To see full help information on how to use preprocess.py script use this command:
   ```bash
    python protocol_finetune.py --help
   ```
   Fine-tune example command:
   ```bash
    python protocol_finetune.py \
     --max_seq_len=5001 \
     --include_zero_gene=True \
     --epochs=3 \
     --batch_size=32 \
     --schedule_ratio=0.9
   ```
   > *--max_seq_len* <= *--n_hvg*
   
3. **Inference** \
   Evaluation and benchmark will be executed by this part. \
   To see full help information on how to use preprocess.py script use this command:
   ```bash
    python protocol_inference.py --help
   ```
   Run inference:
   ```bash
    python protocol_inference.py \
     --load_model=save/dev_eyescGPT_May0520 \
     --batch_size=32 \
     --wandb_sync=True \
     --wandb_project=benchmark_BC \
     --wandb_name=sample_bm_0520
   ```
   
## Useful Hints
* How to use custom config file \
   You can use the custom config file by inserting the path for the variable `--config`. You can see more details in `docs/*-help.txt` \
   Example: 
   ```bash
     python protocol_preprocess.py \
        --dataset_directory=../datasets/retina_snRNA.h5ad \
        --config=save/dev_eyescGPT_May0520/custom_config.yml \   <<<<< Custom Config
        --cell_type_col=celltype \
        --batch_id_col=sampleid \
        --load_model=../scGPT_human \
         --wandb_sync=True \
        --wandb_project=finetune_retina_snRNA \
        --wandb_name=finetune_example1
   ```
  
* Notebooks \
   Notebooks in `/notebooks`
