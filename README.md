# Fine-tune Protocol for eye-scGPT
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14648190.svg)](https://doi.org/10.5281/zenodo.14648190)

This is a protocol for doing fine-tuning on any single-cell annotations `(ex: .h5ad, .hd5f, etc.)` with [scGPT](https://www.nature.com/articles/s41592-024-02201-0).
## Quick How-to-Use Guide
* **Datasets**
  1. Select and download train/eval datasets from [HERE](https://zenodo.org/records/13896150)

* **Fine-tuned eye-scGPT Model**
  1. You can download the fine-tuned eye-scGPT model [HERE](https://zenodo.org/records/13896150/files/finetuned_AiO.zip?download=1) or you can use `curl` to download through terminal:
     ```bash
     curl -L -o finetuned_AiO.zip "https://zenodo.org/api/records/14648190/files/finetuned_AiO.zip"
     ```
 
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
* How to dynamically test the highest batch size for my environment \
    You can use the dry-run mode with a initial batch size to find out the largest possible batch size under your computing environment. \
    Here is how you can use the dry-run mode:
    > NOTE: the default value for initial batch size is set to 32
    ```bash
    python protocol_finetune.py \
        --dry_run=True \
        --batch_size=32 \
        --max_seq_len=5001 \
        --include_zero_gene=True
    ```
* Early-stopping example \
    This is a good practice to prevent overfitting and un-converging training. The following is an example early stopping class that tracks the minimum validation loss value with a allow chance.
    ```python
    # Early-stop class  
    class Protocol_EarlyStop:
        def __init__(self, allow_chance=1, min_delta=0):
            self.allow_chance = allow_chance
            self.min_delta = min_delta
            self.counter = 0
            self.min_validation_loss = float('inf')
        def early_stop(self, validation_loss):
            if validation_loss < self.min_validation_loss:
                self.min_validation_loss = validation_loss
                self.counter = 0
            elif validation_loss > (self.min_validation_loss + self.min_delta):
                self.counter += 1
                if self.counter >= self.allow_chance:
                    return True
            return False
  
    ```
    ```python
    # Eample usage
    # **NOTE** This is NOT the functional code
    early_stopper = Protocol_EarlyStop(allow_chance=3, min_delta=0.1)
    for epoch in np.arange(n_epochs):
        train_loss = train(model, train_loader)
        validation_loss = validate_epoch(model, validation_loader)
        if early_stopper.early_stop(validation_loss):             
            break
    ```
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

## Citation
Ding, S., Li, J., Luo, R. et al. scGPT: end-to-end protocol for fine-tuned retinal cell type annotation. Nat Protoc (2025). https://doi.org/10.1038/s41596-025-01220-1
