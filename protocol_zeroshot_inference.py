#%%
from utils.build_atlas_index_faiss import load_index, vote
from utils.protocol_prelude import *

try:
    import faiss
except ImportError:
    print("faiss is not correctly configured! Please try to install the package by the following command line:\npip install -U faiss-cpu")
    print("To install GPU or other version, see https://github.com/facebookresearch/faiss/wiki/Installing-Faiss")


#%%
@click.command()
@click.option('--scgpt_index_path', type=str, required=True, help='Directory to the scGPT pretrained cell atlas index')
@click.option('--eval_data_path', type=str, required=True, help='Path to the evaluation dataset')
@click.option('--model_path', type=str, required=True, help='Path to a pre-trained scGPT')
@click.option('--cell_type_col', type=str, required=True, help='Column name for ground truth cell type')
@click.option('--k_value', type=int, default=50, help='Column name for ground truth cell type')
def main(
    scgpt_index_path: str,
    eval_data_path: str,
    model_path: str,
    cell_type_col: str,
    k_value: int
):
    warnings.filterwarnings("ignore", category=ResourceWarning)
    job_name = (eval_data_path.split('/')[-1]).split('.')[0]
    use_gpu = faiss.get_num_gpus() > 0
    index, meta_labels = load_index(
        index_dir=scgpt_index_path,
        use_config_file=False,
        use_gpu=use_gpu,
    )
    print(f"Loaded index with {index.ntotal} cells")

    eval_adata = sc.read_h5ad(eval_data_path)
    eval_adata.var = eval_adata.var.set_index("gene_symbols")
    model_dir = Path(model_path)
    gene_col = "index"

    eval_embed_adata = scg.tasks.embed_data(
        eval_adata,
        model_dir,
        gene_col=gene_col,
        batch_size=64,
        device="gpu" if use_gpu else "cpu",
        use_fast_transformer=True if use_gpu else False,
        return_new_adata=True,
    )
    eval_embed = eval_embed_adata.X
    distances, idx = index.search(eval_embed, k_value)
    predict_labels = meta_labels[idx]
    voting = []
    for preds in tqdm(predict_labels):
        voting.append(vote(preds, return_prob=False)[0])
    voting = np.array(voting)

    gt = eval_adata.obs[cell_type_col].to_numpy()
    preds = voting.copy()

    df = pd.DataFrame({
        'ground_truth': gt,
        'prediction': preds
    })

    save_dir = Path(f"./save/zs_{job_name}-{time.strftime('%b%d-%H-%M-%S')}/")
    df.to_csv(save_dir / 'ground_truth_and_predictions.csv', index=False)
    print('=' * 30)
    print(f'Zero-shot prediction is saved to -> {save_dir}')
    print('=' * 30)


#%% Run
if __name__ == '__main__':
    main()


#%% END
