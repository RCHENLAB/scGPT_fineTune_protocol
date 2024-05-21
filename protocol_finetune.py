# %%
from utils import *


@click.command()
@click.option('--max_seq_len', type=int, required=True, help='Max input sequence length during training. The length should be <= n_hvg+1')
@click.option('--config', type=str, required=True, help='Use config file. If using custom config file, input the path to the file directly.  Options=[pp, train, Any<Path>]')
@click.option('--include_zero_gene', type=bool, default=False, help='Include all zero genes in sequence. Default=False')
@click.option('--append_cls', type=bool, default=True, help='Append <cls> to the sequence. Default=True')
@click.option('--epochs', type=int, default=3, help='Epochs for training. Default=3')
@click.option('--use_flash_attn', type=bool, default=False, help='Enable flash-attn. Default=False')
@click.option('--pre_norm', type=bool, default=False, help='Enable pre-normalization. Default=False')
@click.option('--amp', type=bool, default=True, help='Enable automatic mixed precision. Default=True')
@click.option('--freeze_predecoder', type=bool, default=False, help='Freeze pre-decoder. Default=False')
@click.option('--train_test_split_ratio', type=float, default=0.1, help='Ratio for splitting train/val dataset. Default=0.1')
@click.option('--batch_size', type=int, default=32, help='Batch size during training. Default=32')
@click.option('--lr', type=float, default=1e-4, help='Learning rate. Default=0.0001')
@click.option('--dropout', type=float, default=0.2, help='Dropout rate. Default=0.2')
@click.option('--schedule_ratio', type=float, default=0.9, help='Learning rate changing ratio per schedule interval. Default=0.9')
@click.option('--schedule_interval', type=int, default=1, help='Epochs to change the learning rate. Default=1')
@click.option('--save_eval_interval', type=int, default=1, help='Epochs to do evaluation and save the best model so far. Default=1')
@click.option('--nlayers', type=int, default=12, help='Transformer encoder layers. Default=12')
@click.option('--nheads', type=int, default=4, help='Attention heads. Default=4')
@click.option('--embsize', type=int, default=512, help='Embedding dimension. Default=512')
@click.option('--nlayers_cls', type=int, default=3, help='Decoder layers for classifier. Default=3')
def main(
    max_seq_len,
    config,
    include_zero_gene,
    append_cls,
    epochs,
    use_flash_attn,
    pre_norm,
    amp,
    freeze_predecoder,
    train_test_split_ratio,
    batch_size,
    lr,
    dropout,
    schedule_ratio,
    schedule_interval,
    nlayers,
    nheads,
    embsize,
    nlayers_cls
):
    # training configs
    if config == 'pp':
        hyperparameter_defaults = load_config(preprocess=True)
    elif config == 'train':
        hyperparameter_defaults = load_config(train=True)
    else:
        hyperparameter_defaults = load_config(custom_config=config)

    task_info = hyperparameter_defaults['task_info']
    preprocess_config = hyperparameter_defaults['preprocess']
    model_params = hyperparameter_defaults['model_parameters']
    task_configs = hyperparameter_defaults['task_configs']

    # update configs
    model_params['epochs'] = epochs
    model_params['use_flash_attn'] = use_flash_attn
    model_params['pre_norm'] = pre_norm
    model_params['amp'] = amp
    model_params['freeze_predecoder'] = freeze_predecoder
    model_params['train_test_split_ratio'] = train_test_split_ratio
    model_params['batch_size'] = batch_size
    model_params['lr'] = lr
    model_params['dropout'] = dropout
    model_params['schedule_ratio'] = schedule_ratio
    model_params['schedule_interval'] = schedule_interval
    model_params['nlayers'] = nlayers
    model_params['nheads'] = nheads
    model_params['embsize'] = embsize
    model_params['nlayers_cls'] = nlayers_cls
    task_configs['include_zero_gene'] = include_zero_gene
    task_configs['append_cls'] = append_cls

    # Create WandB instance and enable cloud-syncing
    wandb_instance = ProtocolWandB(hyperparameter_defaults)
    run = wandb_instance.create_wandb_project()
    config = wandb.config  # easy-dict access

    # create local save directory for fine-tuning and save updated config file
    set_seed(model_params['seed'])
    dataset_name = task_info['raw_dataset_name']
    save_dir = Path(f"./save/dev_{dataset_name}-{time.strftime('%b%d-%H-%M-%S')}/")
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = scg.logger
    scg.utils.add_file_handler(logger, save_dir / "run.log")
    os.system(f"cp {__file__} {save_dir}")
    with open(save_dir / 'dev_train_args.yml', 'w') as out_configs:
        yaml.dump(hyperparameter_defaults, out_configs, sort_keys=False)
    logger.info(f"Current training script and config file is saved to -> {save_dir}")
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
    input_style = config.task_configs['input_style']

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

    ## sanity check
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

    # Read input annData
    adata = sc.read_h5ad(config.task_info['input_dataset_directory'], backed='r')
    num_types = len(np.unique(adata.obs[config.preprocess['_FIXED_CELL_TYPE_COL']]))
    genes = adata.var[config.preprocess['_FIXED_GENE_COL']].tolist()

    # Load pre-trained model weights
    logger.info(f'Start loading pre-trained model and weights ...')
    model_dir = Path(config.model_parameters['load_model'])
    model_file = model_dir / "best_model.pt"  # TODO: change the model file name if different
    model_config_file = model_dir / "args.json"
    vocab_file = model_dir / "vocab.json"
    vocab = get_vocab(vocab_file, special_tokens)
    vocab.set_default_index(vocab["<pad>"])
    logger.info(
        f"Resume model from {model_file}, the model args will be override by the "
        f"config {model_config_file}."
    )

    # Create train and validation datasets
    gene_ids = np.array(vocab(genes), dtype=int)
    indices = list(range(adata.shape[0]))
    train_indices, val_indices = train_test_split(indices, test_size=train_test_split_ratio, random_state=42)
    train_dataset = Subset(SeqDataset(adata), train_indices)
    val_dataset = Subset(SeqDataset(adata), val_indices)

    # Create data collator and loaders
    data_collator = DataCollator(
        do_padding=True,
        pad_token_id="<pad>",
        do_mlm=False,
        do_binning=True,
        mask_value=mask_value,
        pad_value=-2,
        max_length=max_seq_len,
        data_style="cls",
        filtered_gene_list=gene_ids,
        vocab=vocab,
        append_cls=append_cls,
        include_zero_gene=include_zero_gene
    )

    train_sampler = (RandomSampler(train_dataset))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=data_collator,
        drop_last=False,
        num_workers=0,  # !!! Enable data stream, ONLY 1 process
        pin_memory=True
    )

    valid_sampler = (SequentialSampler(val_dataset))
    valid_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
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

    pre_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())

    # Freeze all pre-decoder weights
    for name, para in model.named_parameters():
        if config.model_parameters['freeze_predecoder'] and "encoder" in name and "transformer_encoder" not in name:
            # if config.freeze and "encoder" in name:
            print(f"freezing weights for: {name}")
            para.requires_grad = False

    post_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())

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

    # Start fine-tuning
    logger.info(f'Start training ...')
    best_val_loss = float("inf")
    best_model = None
    wandb_instance.define_wandb_metrcis()

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()

        if config.model_parameters['do_train']:
            train(
                model,
                loader=train_loader,
                pad_vocab=vocab[pad_token],
                criterion_cls=criterion_cls,
                config=config,
                scheduler=scheduler,
                scaler=scaler,
                optimizer=optimizer,
                epoch=epoch,
                log_interval=log_interval,
                logger=logger,
                is_parallel=is_parallel,
                device=device
            )
        val_loss, val_err = evaluate(
            model,
            loader=valid_loader,
            pad_vocab=vocab[pad_token],
            criterion_cls=criterion_cls,
            config=config,
            epoch=epoch,
            device=device
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

        if epoch % save_eval_interval == 0 or epoch == config.model_parameters['epochs']:
            logger.info(f"Saving model to {save_dir}")
            torch.save(best_model.state_dict(), save_dir / f"model_e{best_model_epoch}.pt")

        scheduler.step()

    # Terminate task
    torch.save(best_model.state_dict(), save_dir / "best_model.pt")
    artifact = wandb.Artifact(f"best_model", type="model")
    glob_str = os.path.join(save_dir, "best_model.pt")
    artifact.add_file(glob_str)
    run.log_artifact(artifact)
    logger.info(f'Training job is saved to {save_dir}')

    run.finish()
    wandb.finish()
    gc.collect()


#%% Run
if __name__ == '__main__':
    main()


#%% END
