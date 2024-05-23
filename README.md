# Fine-tune Protocol for eye-scGPT

`Maintainer of the protocol: Shanli Ding`

This is a protocol for doing fine-tuning on any single-cell dataset `(ex: .h5ad, .hd5f, etc.)` with [scGPT](https://www.nature.com/articles/s41592-024-02201-0).

## How-to-Use
1. **Pre-process** \
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
     --batch_id_col=sampleid \
     --load_model=../scGPT_human \
      --wandb_sync=True \
     --wandb_project=finetune_retina_snRNA \
     --wandb_name=finetune_example1
   ```
   > Please keep in mind to change variable's values to your own settings

2. **Fine-tune** \
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
     --include_zero_gene=True \
     --epochs=3 \
     --batch_size=32 \
     --schedule_ratio=0.9
   ```
   
3. **Inference** \
   Evaluation and benchmark will be executed by this part. \
   To see full help information on how to use preprocess.py script use this command:
   ```bash
    python protocol_inference.py --help
   ```
   Run inference:
   ```bash
    python protocol_inference.py \
     --load_model=save/dev_eyescGPT_May0520
     --batch_size=32 \
     --wandb_sync=True \
     --wandb_project=benchmark_BC \
     --wandb_name=sample_bm_0520
   ```