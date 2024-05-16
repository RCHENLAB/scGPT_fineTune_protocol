# %%
from utils import *


#%% Read fine-tune configs
with open('train_args.yml', 'r') as in_configs:
    hyperparameter_defaults = yaml.safe_load(in_configs)
set_seed(hyperparameter_defaults['model_parameters']['seed'])

#%% Create WandB instance and enable cloud-syncing
wandb_instance = ProtocolWandB(hyperparameter_defaults)
run = wandb_instance.create_wandb_project()
config = wandb.config  # easy-dict access


#%% Initialize directory for fine-tuning
dataset_name = config.dataset['name']
save_dir = Path(f"./save/dev_{dataset_name}-{time.strftime('%b%d-%H-%M-%S')}/")
save_dir.mkdir(parents=True, exist_ok=True)
logger = scg.logger
scg.utils.add_file_handler(logger, save_dir / "run.log")
os.system(f"cp {__file__} {save_dir}")
logger.info(f"Current training script is saved to -> {save_dir}")
logger.info(f"Working directory is initialized successfully ...")
print()


#%% constants
pad_token = config.task_configs['pad_token']
special_tokens = [pad_token, config.task_configs['cls_token'], config.task_configs['eoc_token']]
mask_value = config.task_configs['mask_value']
pad_value = config.task_configs['pad_value']
include_zero_gene = config.task_configs['include_zero_gene']
max_seq_len = config.task_configs['max_seq_len']
n_input_bins = config.task_configs['n_input_bins']
input_style = config.task_configs['input_style']

# Task configurations
CLS = config.task_configs['CLS']
input_emb_style = config.task_configs['input_emb_style']
cell_emb_style = config.task_configs['cell_emb_style']
ecs_threshold = config.task_configs['ecs_threshold']

# Model Parameters
lr = config.model_parameters['lr']
batch_size = config.model_parameters['batch_size']
eval_batch_size = config.model_parameters['batch_size']
epochs = config.model_parameters['epochs']
schedule_interval = config.model_parameters['schedule_interval']
fast_transformer = config.model_parameters['use_flash_attn']
fast_transformer_backend = config.model_parameters['fast_transformer_backend']
embsize = config.model_parameters['embsize']
d_hid = config.model_parameters['d_hid']
nlayers = config.model_parameters['nlayers']
nhead = config.model_parameters['nhead']
dropout = config.model_parameters['dropout']
log_interval = config.model_parameters['log_interval']
save_eval_interval = config.model_parameters['save_eval_interval']

assert input_style in ["normed_raw", "log1p", "binned"]
assert input_emb_style in ["category", "continuous", "scaling"]
if input_style == "binned":
    if input_emb_style == "scaling":
        raise ValueError("input_emb_style `scaling` is not supported for binned input.")
elif input_style == "log1p" or input_style == "normed_raw":
    if input_emb_style == "category":
        raise ValueError(
            "input_emb_style `category` is not supported for log1p or normed_raw input."
        )


#%% Read input annData
adata = sc.read_h5ad(config.task_info['input_dataset_directory'], backed='r')
num_types = len(np.unique(adata.obs[config.preprocess[['_FIXED_CELL_TYPE_COL']]]))
genes = adata.var[config.preprocess['_FIXED_GENE_COL']]

#%% Loading pre-trained model weights
logger.info(f'Start loading pre-trained model and weights ...')
model_dir = Path(config.load_model)
model_file = model_dir / "best_model.pt"
model_config_file = model_dir / "args.json"
vocab_file = model_dir / "vocab.json"
vocab = get_vocab(vocab_file, special_tokens)
vocab.set_default_index(vocab["<pad>"])
logger.info(
    f"Resume model from {model_file}, the model args will be override by the "
    f"config {model_config_file}."
)
embsize = config.model_parameters['embsize']
nhead = config.model_parameters['nhead']
d_hid = config.model_parameters['d_hid']
nlayers = config.model_parameters['nlayers']
nlayers_cls = config.model_parameters['nlayers_cls']


#%%
gene_ids = np.array(vocab(genes), dtype=int)
indices = list(range(adata.shape[0]))
train_indices, test_indices = train_test_split(indices, test_size=0.1, random_state=42)

# Create train and test subsets
train_dataset = Subset(SeqDataset(adata), train_indices)
test_dataset = Subset(SeqDataset(adata), test_indices)

#%% END
