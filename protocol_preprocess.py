# %%
from utils import *


#%% Load config file
with open('./utils/basic_train_args.yml', 'r') as in_configs:
    hyperparameter_defaults = yaml.safe_load(in_configs)

task_info = hyperparameter_defaults['task_info']
preprocess_config = hyperparameter_defaults['preprocess']
model_params = hyperparameter_defaults['model_parameters']


# %%
dataset_name = task_info['raw_dataset_name']
save_dir = Path(f"./save/pp_{dataset_name}-{time.strftime('%b%d-%H-%M-%S')}/")
save_dir.mkdir(parents=True, exist_ok=True)
logger = scg.logger
task_info['save_dir'] = str(save_dir)
logger.info(f"Running pre-process program is saved to -> {save_dir}")


#%%
file_dir = task_info['raw_dataset_directory']
logger.info(f'Start reading data from {file_dir}...')
adata = sc.read_h5ad(file_dir)
_cell_type_col = preprocess_config['_FIXED_CELL_TYPE_COL']
_cell_type_id = preprocess_config['_FIXED_CELL_TYPE_ID_COL']
_gene_name = preprocess_config['_FIXED_GENE_COL']
_batch_id = preprocess_config['_FIXED_BATCH_ID_COL']
input_data_cell_type_col = preprocess_config['dataset_cell_type_col']
input_data_batch_id_col = preprocess_config['dataset_batch_id_col']
adata.obs[_cell_type_col] = adata.obs[input_data_cell_type_col].astype("category")
adata.obs[_batch_id] = adata.obs[input_data_batch_id_col].astype("category").cat.codes.values
celltype_id_labels = adata.obs[_cell_type_col].astype("category").cat.codes.values
adata.obs[_cell_type_id] = celltype_id_labels
adata.var[_gene_name] = adata.var.index.tolist()


#%% Load and save pre-trained model settings
load_model = hyperparameter_defaults['model_parameters']['load_model']
model_dir = Path(load_model)
model_config_file = model_dir / "args.json"
model_file = model_dir / "best_model.pt"
vocab_file = model_dir / "vocab.json"
shutil.copy(vocab_file, save_dir / "vocab.json")
vocab = GeneVocab.from_file(vocab_file)
pad_token = hyperparameter_defaults['task_configs']['pad_token']
special_tokens = [pad_token, hyperparameter_defaults['task_configs']['cls_token'], hyperparameter_defaults['task_configs']['eoc_token']]
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

(
    model_params['embsize'],
    model_params['nheads'],
    model_params['d_hid'],
    model_params['nlayers'],
    model_params['nlayers_cls'],
) = (
    model_configs["embsize"],
    model_configs["nheads"],
    model_configs["d_hid"],
    model_configs["nlayers"],
    model_configs["n_layers_cls"],
)


#%% Pre-processing
preprocessor = Preprocessor(
    logger,
    use_key="X",  # layer name for count matrix
    filter_gene_by_counts=preprocess_config['filter_gene_by_counts'],  # step 1
    filter_cell_by_counts=preprocess_config['filter_cell_by_counts'],  # step 2
    normalize_total=preprocess_config['normalize_total'],  # 3. whether to normalize the raw data and to what sum
    log1p=preprocess_config['do_norm'],  # 4. whether to log1p the normalized data
    subset_hvg=preprocess_config['n_hvg'],  # 5. whether to subset the raw data to highly variable genes
    hvg_flavor=preprocess_config['hvg_flavor'],
)
logger.info(f"Original data shape: {adata.shape}")
preprocessor(adata, batch_key='batch_id')
logger.info(f"Preprocessed data shape: {adata.shape}")
logger.info(f'Saving processed adata to {save_dir} ...')
input_dataset_directory = save_h5ad(adata, save_dir)
task_info['input_dataset_directory'] = input_dataset_directory
with open('train_args.yml', 'w') as out_configs:
    yaml.dump(hyperparameter_defaults, out_configs, sort_keys=False)
logger.info('=' * 20)
logger.info(f'Saved training-ready config file to current directory ...')
logger.info('Pre-processing finished. Cleaning cache ... Terminating ...')
logger.info('=' * 20)
print()
gc.collect()


#%% END
