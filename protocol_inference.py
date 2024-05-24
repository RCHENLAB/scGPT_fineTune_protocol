# %%
from utils import *


@click.command()
@click.option('--load_model', type=str, required=True, help='directory to pretrained/tuned model directory.')
@click.option('--max_seq_len', type=int, default=-1, help='Max input sequence length during training. The length should be <= n_hvg+1. Default=same length as loaded model definition')
@click.option('--config', type=str, default='eval', help='Use config file. If using custom config file, input the path to the file directly.  Options=[pp, train, eval, Any<Path>]')
@click.option('--freeze_predecoder', type=bool, default=False, help='Freeze pre-decoder. Default=False')
@click.option('--batch_size', type=int, default=32, help='Batch size during evaluation. Default=32')
@click.option('--wandb_sync', type=bool, default=False, help='Enable WandB cloud syncing. Default=False')
@click.option('--wandb_project', type=str, required=True, help='Project name in WandB. Recommend to use different project name other than training project.')
@click.option('--wandb_name', type=str, default='', help='Run name in WandB. Default=EMPTY.')
def main(
    load_model,
    max_seq_len,
    config,
    freeze_predecoder,
    batch_size,
    wandb_sync,
    wandb_project,
    wandb_name,
    epochs=1
):
    # get loaded model configs
    if config == 'pp':
        hyperparameter_defaults = load_config(preprocess=True)
    elif config == 'train':
        hyperparameter_defaults = load_config(train=True)
    elif config == 'eval':
        hyperparameter_defaults = load_config(eval=True, load_model_dir=load_model)
    else:
        hyperparameter_defaults = load_config(custom_config=config)

    # update configs
    task_info = hyperparameter_defaults['task_info']
    preprocess_config = hyperparameter_defaults['preprocess']
    model_params = hyperparameter_defaults['model_parameters']
    task_configs = hyperparameter_defaults['task_configs']
    wandb_config = hyperparameter_defaults['wandb_configs']

    model_params['load_model'] = load_model
    model_params['epochs'] = epochs
    model_params['batch_size'] = batch_size
    model_params['freeze_predecoder'] = freeze_predecoder
    task_configs['max_seq_len'] = task_configs['max_seq_len'] if max_seq_len == -1 else max_seq_len
    wandb_config['mode'] = 'online' if wandb_sync else 'offline'
    wandb_config['project'] = wandb_project
    wandb_config['name'] = wandb_name

    # Create WandB instance and enable cloud-syncing
    wandb_instance = ProtocolWandB(hyperparameter_defaults)
    run = wandb_instance.create_wandb_project()
    config = wandb.config  # easy-dict access

    # START TIME POINT
    START_POINT = time.time()

    # create local save directory for fine-tuning and save updated config file
    set_seed(model_params['seed'])
    dataset_name = task_info['raw_dataset_name']
    save_dir = Path(f"./save/eval_{dataset_name}-{time.strftime('%b%d-%H-%M-%S')}/")
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = scg.logger
    scg.utils.add_file_handler(logger, save_dir / "run.log")
    logger.info(f"Current evaluation progress is saved to -> {save_dir}")
    logger.info(f"Working directory is initialized successfully ...")
    print()

    # CONSTANTS
    pad_token = config.task_configs['pad_token']
    special_tokens = [pad_token, config.task_configs['cls_token'], config.task_configs['eoc_token']]
    mask_value = config.task_configs['mask_value']
    pad_value = config.task_configs['pad_value']
    append_cls = config.task_configs['append_cls']
    include_zero_gene = config.task_configs['include_zero_gene']
    max_seq_len = config.task_configs['max_seq_len']
    n_input_bins = config.task_configs['n_input_bins']
    _cell_type_col = preprocess_config['_FIXED_CELL_TYPE_COL']
    _cell_type_id = preprocess_config['_FIXED_CELL_TYPE_ID_COL']

    ## Task configurations
    CLS = config.task_configs['CLS']
    input_emb_style = config.task_configs['input_emb_style']
    cell_emb_style = config.task_configs['cell_emb_style']
    ecs_threshold = config.task_configs['ecs_threshold']

    ## Model Parameters
    lr = config.model_parameters['lr']
    batch_size = config.model_parameters['batch_size']
    epochs = config.model_parameters['epochs']
    schedule_interval = config.model_parameters['schedule_interval']
    fast_transformer = config.model_parameters['use_flash_attn']
    fast_transformer_backend = config.model_parameters['fast_transformer_backend']
    embsize = config.model_parameters['embsize']
    d_hid = config.model_parameters['d_hid']
    nlayers = config.model_parameters['nlayers']
    nhead = config.model_parameters['nheads']
    nlayers_cls = config.model_parameters['nlayers_cls']
    dropout = config.model_parameters['dropout']
    log_interval = config.model_parameters['log_interval']
    save_eval_interval = config.model_parameters['save_eval_interval']

    # Read input annData
    adata = sc.read_h5ad(config.task_info['input_dataset_directory'], backed='r')
    num_types = len(np.unique(adata.obs[config.preprocess['_FIXED_CELL_TYPE_COL']]))
    genes = adata.var[config.preprocess['_FIXED_GENE_COL']].tolist()

    # Load pre-trained model weights
    logger.info(f'Start loading pre-trained model and weights ...')
    model_dir = Path(model_params['load_model'])
    model_file = model_dir / "best_model.pt"  # TODO: change the model file name if different
    model_config_file = model_dir / "args.json"
    vocab_file = model_dir / "vocab.json"
    vocab = get_vocab(vocab_file, special_tokens)
    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(vocab(genes), dtype=int)
    logger.info(
        f"Resume model from {model_file}, the model args will be override by the "
        f"config {model_config_file}."
    )

    # Create data collator and loaders
    data_collator = DataCollator(
        do_padding=True,
        pad_token_id="<pad>",
        do_mlm=False,
        do_binning=True,
        mask_value=mask_value,
        pad_value=pad_value,
        max_length=max_seq_len,
        data_style="cls",
        filtered_gene_list=gene_ids,
        vocab=vocab,
        append_cls=append_cls,
        include_zero_gene=include_zero_gene
    )

    test_dataset = SeqDataset(adata)
    test_sampler = (SequentialSampler(test_dataset))
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        collate_fn=data_collator,
        drop_last=False,
        num_workers=0,  # !!! Enable data stream, ONLY 1 process
        pin_memory=True
    )

    # Create model instance
    logger.info(f'Create new model instance ...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ntokens = len(vocab)
    model = TransformerModel(
        ntokens,
        embsize,
        nhead,
        d_hid,
        nlayers,
        nlayers_cls=nlayers_cls,
        n_cls=num_types if CLS else 1,
        vocab=vocab,
        dropout=dropout,
        pad_token=pad_token,
        pad_value=pad_value,
        do_mvc=config.task_configs['MVC'],
        do_dab=config.task_configs['DAB'],
        use_batch_labels=config.model_parameters['use_batch_labels'],
        domain_spec_batchnorm=config.model_parameters['DSBN'],
        input_emb_style=input_emb_style,
        n_input_bins=n_input_bins,
        cell_emb_style=cell_emb_style,
        ecs_threshold=ecs_threshold,
        use_fast_transformer=fast_transformer,
        fast_transformer_backend=fast_transformer_backend,
        pre_norm=config.model_parameters['pre_norm'],
    )
    if config.model_parameters['load_model'] is not None:
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
            # for k, v in pretrained_dict.items():
            #     logger.info(f"Loading params {k} with shape {v.shape}")
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

    pre_freeze_param_count = sum(
        dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())

    # Freeze all pre-decoder weights
    for name, para in model.named_parameters():
        if config.model_parameters['freeze_predecoder'] and "encoder" in name and "transformer_encoder" not in name:
            # if config.freeze and "encoder" in name:
            print(f"freezing weights for: {name}")
            para.requires_grad = False

    post_freeze_param_count = sum(
        dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())

    logger.info(f"Total Pre freeze Params {(pre_freeze_param_count)}")
    logger.info(f"Total Post freeze Params {(post_freeze_param_count)}")

    model.to(device)
    is_parallel = torch.cuda.device_count() > 1
    if is_parallel:
        model = nn.DataParallel(model)

    wandb.watch(model)

    # Define loss, optimizer, scaler
    logger.info(f'Define loss metrics and optimizer ...')
    ################################################
    # # Default Cross-entropy
    criterion_cls = nn.CrossEntropyLoss()
    ################################################
    # TODO: Weighted loss
    # # Weighted Cross-entropy
    # # Normalized Inverse Class Frequency
    # class_counts = np.bincount(celltype_id_labels)
    # class_weights = len(celltype_id_labels) / (len(class_counts) * class_counts)
    # criterion_cls = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float, device=device))
    ################################################
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, eps=1e-4 if config.model_parameters['amp'] else 1e-8
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, schedule_interval, gamma=config.model_parameters['schedule_ratio']
    )
    scaler = torch.cuda.amp.GradScaler(enabled=config.model_parameters['amp'])

    logger.info(f'Start evaluating ... ')
    with open(config.task_info['id2type_json'], 'r') as f:
        ref_id2type = json.load(f)
    unique_ref_cell_types = list(ref_id2type.values())
    labels = adata.obs[_cell_type_id].tolist()
    test_cell_types = adata.obs[_cell_type_col].unique()
    (
        predictions,
        results,
        precision_dict,
        wrong_predictions
    ) = test(
        model,
        loader=test_loader,
        adata=adata,
        true_cell_type_ids=labels,
        ref_id2type=ref_id2type,
        pad_vocab=vocab[pad_token],
        criterion_cls=criterion_cls,
        config=config,
        epoch=epochs,
        logger=logger,
        device=device
    )

    # END TIME POINT
    END_POINT = time.time()
    inference_time = round(END_POINT - START_POINT, 2)
    wandb.log(
        {
            "Evaluation Time": inference_time
        }
    )
    logger.info(f'*** Evaluation was finished in: {inference_time} seconds ***')

    # generate UMAP
    logger.info(f'Generating UMAP plots ... ')
    adata.obs["predictions"] = [ref_id2type[p] for p in predictions]

    palette_ = []
    while len(palette_) < len(unique_ref_cell_types):
        palette_ += plt.rcParams["axes.prop_cycle"].by_key()["color"]
    palette_ = {c: palette_[i] for i, c in enumerate(unique_ref_cell_types)}

    sc.tl.pca(adata)
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30)
    sc.tl.umap(adata)

    with plt.rc_context({"figure.dpi": 600}):
        sc.pl.umap(
            adata,
            color=["celltype", "predictions"],
            palette=palette_,
            show=False,
            legend_fontsize=6,
            wspace=.5
        )
        plt.savefig(save_dir / "results.png", dpi=600, bbox_inches="tight")

    save_dict = {
        "predictions": predictions,
        "labels": labels,
        "results": results,
        "id_maps": ref_id2type
    }

    with open(save_dir / "results.pkl", "wb") as f:
        pickle.dump(save_dict, f)

    results["test/cell_umap"] = wandb.Image(
        str(save_dir / "results.png"),
        caption=f"predictions macro f1 {results['test/macro_f1']:.3f}",
    )
    wandb.log(results)

    # generate confusion matrix
    logger.info(f'Generating confusion matrix ... ')

    cm = confusion_matrix(adata.obs[_cell_type_col], [ref_id2type[p] for p in predictions], labels=test_cell_types)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    prediction_result_dict = {
        "precision_per_label": precision_dict,
        "error": wrong_predictions
    }

    with open(save_dir / "evaluation_results.json", "w") as json_file:
        json.dump(prediction_result_dict, json_file, indent=4)
        print(json.dumps(prediction_result_dict, indent=4))

    print("Evaluation results saved to 'evaluation_results.json'")

    plt.figure(figsize=(20, 20))
    sns.heatmap(cm_normalized, annot=True, cmap='plasma', xticklabels=test_cell_types, yticklabels=test_cell_types)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(save_dir / "confusion_matrix.png", dpi=600)

    results["test/confusion_matrix"] = wandb.Image(
        str(save_dir / "confusion_matrix.png"),
        caption=f"confusion matrix",
    )

    run.finish()
    wandb.finish()
    gc.collect()
    logger.info(f'Evaluation is saved to directory ==> {save_dir}')
    logger.info(f'Benchmarking was completed ! Well done =) ')


#%% Run
if __name__ == '__main__':
    main()


#%% END
