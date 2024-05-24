#%%
from .protocol_prelude import *


#%% Train func
def train(
    model: nn.Module,
    loader: DataLoader,
    pad_vocab: int,  # =
    criterion_cls,
    config,
    scheduler,
    scaler,
    optimizer,
    epoch,
    log_interval,
    logger,
    device,
    is_parallel=False,
    batch_labels=None,
) -> None:
    """
    Train the model for one epoch.
    """
    model.train()
    (total_loss, total_cls) = (0.0, 0.0)
    CLS = config.task_configs['CLS']
    CCE = config.task_configs['CCE']
    MVC = config.task_configs['MVC']
    ECS = config.task_configs['ECS']

    total_error = 0.0
    start_time = time.time()
    num_batches = len(loader)
    for batch, batch_data in enumerate(loader):
        batch_data = {k: v.to(device) for k, v in batch_data.items()}
        input_gene_ids = batch_data["input_gene_ids"].to(device)
        input_values = batch_data["input_expr"].to(device)
        celltype_labels = batch_data["cell_types"].to(device)

        src_key_padding_mask = input_gene_ids.eq(pad_vocab)
        with torch.cuda.amp.autocast(config.model_parameters['amp']):
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels,
                CLS=CLS,
                CCE=CCE,
                MVC=MVC,
                ECS=ECS,
                do_sample=config.task_configs['do_sample_in_train']
            )

            loss = 0.0
            if CLS:
                loss_cls = criterion_cls(output_dict["cls_output"], celltype_labels)
                metric_loss_cls = torch.mean(torch.sum(loss_cls)) if is_parallel else loss_cls
                loss = loss + metric_loss_cls
                error_rate = 1 - (
                    (output_dict["cls_output"].argmax(1) == celltype_labels)
                    .sum()
                    .item()
                ) / celltype_labels.size(0)

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
                error_if_nonfinite=False if scaler.is_enabled() else True,
            )
            if len(w) > 0:
                logger.warning(
                    f"Found infinite gradient. This may be caused by the gradient "
                    f"scaler. The current scale is {scaler.get_scale()}. This warning "
                    "can be ignored if no longer occurs after autoscaling of the scaler."
                )
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_cls += loss_cls.item()
        total_error += error_rate
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_cls = total_cls / log_interval
            cur_error = total_error / log_interval
            logger.info(
                f"|\tepoch\t{epoch:3d}\t|\t{batch:3d}/{num_batches:3d} batches\t|\t"
                f"lr\t{lr:.4e}\t|\tms/batch\t{ms_per_batch:5.2f}\t|\t"
                f"loss\t{cur_loss:5.2f}\t|\t"
                + (f"cls\t{cur_cls:5.2f}\t|\t")
                + (f"err\t{cur_error:5.2f}\t|\t")
            )
            total_loss = 0
            total_cls = 0
            total_error = 0

            wandb.log(
                {
                    "train/mse": cur_loss,
                    "train/err": cur_error
                },
            )

            start_time = time.time()


#%% Eval func
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    pad_vocab: int,  # = vocab[pad_token]
    criterion_cls,
    config,
    epoch,
    device,
    batch_labels=None,
    return_raw=False,
) -> Union[ndarray[Any, dtype[Any]], Tuple[float, float]]:
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    total_loss = 0.0
    total_error = 0.0
    total_num = 0
    predictions = []
    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["input_gene_ids"].to(device)
            input_values = batch_data["input_expr"].to(device)
            celltype_labels = batch_data["cell_types"].to(device)

            src_key_padding_mask = input_gene_ids.eq(pad_vocab)
            with torch.cuda.amp.autocast(enabled=config.model_parameters['amp']):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels,
                    CLS=config.task_configs['CLS'],
                    CCE=False,
                    MVC=False,
                    ECS=False,
                    do_sample=config.task_configs['do_sample_in_train']
                )
                output_values = output_dict["cls_output"]
                loss = criterion_cls(output_values, celltype_labels)

            total_loss += loss.item() * len(input_gene_ids)
            accuracy = (output_values.argmax(1) == celltype_labels).sum().item()
            total_error += (1 - accuracy / len(input_gene_ids)) * len(input_gene_ids)
            total_num += len(input_gene_ids)
            preds = output_values.argmax(1).cpu().numpy()
            predictions.append(preds)

    wandb.log(
        {
            "valid/mse": total_loss / total_num,
            "valid/err": total_error / total_num,
            "epoch": epoch,
        },
    )

    if return_raw:
        return np.concatenate(predictions, axis=0)

    return total_loss / total_num, total_error / total_num


#%% Test func
def test(
    model: nn.Module,
    loader: DataLoader,
    adata: AnnData,
    true_cell_type_ids: List[str],
    ref_id2type: dict,
    pad_vocab,
    criterion_cls,
    config,
    epoch,
    logger,
    device
):
    model.eval()
    predictions = evaluate(
        model,
        loader=loader,
        pad_vocab=pad_vocab,
        criterion_cls=criterion_cls,
        config=config,
        epoch=epoch,
        device=device,
        return_raw=True
    )
    unique_true_cell_type_ids = [str(i) for i in set(true_cell_type_ids)]
    accuracy = accuracy_score(true_cell_type_ids, predictions)
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(true_cell_type_ids, predictions, labels=unique_true_cell_type_ids)
    macro_precision = precision_per_class.mean()
    macro_recall = recall_per_class.mean()
    macro_f1 = f1_per_class.mean()

    precision_dict = {}
    for label, precision in zip(unique_true_cell_type_ids, precision_per_class):
        precision_dict[ref_id2type[label]] = precision

    wrong_predictions = {}
    for gt, pred, idx in zip(true_cell_type_ids, predictions, range(len(predictions))):
        if gt != pred:
            wrong_predictions[adata.obs.index[idx]] = [ref_id2type[gt], ref_id2type[str(pred)]]

    print('*' * 20)
    logger.info(
        f"Accuracy: {accuracy:.3f}\n"
        f"Precision: {macro_precision:.3f}\n"
        f"Recall: {macro_recall:.3f}\n"
        f"Macro F1: {macro_f1:.3f}"
    )
    print('*' * 20)

    results = {
        "test/accuracy": accuracy,
        "test/precision": macro_precision,
        "test/recall": macro_recall,
        "test/macro_f1": macro_f1,
    }

    return predictions, results, precision_dict, wrong_predictions


#%% END
