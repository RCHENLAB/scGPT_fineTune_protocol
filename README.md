# Fine-tune Protocol for eye-scGPT

This is a protocol for doing fine-tuning on any single-cell dataset `(ex: .h5ad, .hd5f, etc.)` with [scGPT](https://www.nature.com/articles/s41592-024-02201-0).

## How-to-Use
1. Pre-process
   Prepare the custom dataset to train-ready state for next fine-tuning step. \
   To see full help information on how to use preprocess.py script use this command:
   ```bash
    python protocol_preprocess.py --help
   ```
   Run pre-process:
   ```bash
    python protocol_preprocess.py \
     --dataset_directory=../datasets/retina_snRNA.h5ad \
     --cell_type_col=celltype \
     --batch_id_col=batch_id \
     --load_model=../scGPT_human
   ```
   > Please keep in mind to change variable's values to your own settings

2. Fine-tune
   Start fine-tuning the foundation scGPT model with your custom dataset. Here we are introducing our eye-scGPT that is trained specific on
   the human retina single-nuclei and single-cell datasets. Please adjust any parameters with your own requirements. \
   To see full help information on how to use preprocess.py script use this command:
   ```bash
    python protocol_finetune.py --help
   ```
   Run fine-tune:
   ```bash
    python protocol_finetune.py \
     --max_seq_len=5001 \
     --config=train \
     --include_zero_gene=True \
     --epochs=1 \
     --batch_size=32 \
     --schedule_ratio=0.9
   ```