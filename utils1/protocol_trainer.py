#%%
from .protocol_prelude import *


#%% Train func
def train(
    model: nn.Module,
    loader: DataLoader,
    pad_vocab: int,
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


#%% Dry-run
def dry_run_train(
        model: nn.Module,
        loader: DataLoader,
        pad_vocab: int,
        criterion_cls,
        config,
        scaler,
        logger,
        device,
        initial_batch_size=64,
        batch_labels=None,
) -> int:
    """
    Dry-run to determine the largest feasible batch size for training.
    """
    batch_size = initial_batch_size
    while batch_size > 16:
        try:
            # Update loader with the current batch size
            loader.batch_size = batch_size

            logger.info(f"Current batch size: {batch_size}")

            # Perform a dry-run using the first batch
            model.train()
            batch_data = next(iter(loader))  # Get the first batch
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
                    CLS=config.task_configs['CLS'],
                    CCE=config.task_configs['CCE'],
                    MVC=config.task_configs['MVC'],
                    ECS=config.task_configs['ECS'],
                    do_sample=config.task_configs['do_sample_in_train']
                )
                loss = criterion_cls(output_dict["cls_output"], celltype_labels)

            # Backward pass to test memory consumption
            scaler.scale(loss).backward()
            logger.info(f"Batch size {batch_size} is feasible.")
            return batch_size  # Exit if successful

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.warning(f"Batch size {batch_size} failed due to memory error.")
                batch_size -= 1  # Decrease batch size and retry
                torch.cuda.empty_cache()  # Clear memory
            else:
                raise e  # Re-raise other exceptions

    logger.error("No feasible batch size found. Please reduce model size or adjust resources.")
    return 0  # Return 0 if no feasible batch size is found


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
    confidences = []
    is_inference_task = True
    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["input_gene_ids"].to(device)
            input_values = batch_data["input_expr"].to(device)
            celltype_labels = batch_data["cell_types"].to(device)

            src_key_padding_mask = input_gene_ids.eq(pad_vocab)
            with torch.cuda.amp.autocast(enabled=config.model_parameters['amp']):
                output_dict = model(
                    input_gene_ids,
                    input_values.to(torch.float32),
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels,
                    CLS=config.task_configs['CLS'],
                    CCE=False,
                    MVC=False,
                    ECS=False,
                    do_sample=config.task_configs['do_sample_in_train']
                )
                output_values = output_dict["cls_output"]

            if len(set(celltype_labels.tolist())) > 1:
                is_inference_task = False
                loss = criterion_cls(output_values, celltype_labels)
                total_loss += loss.item() * len(input_gene_ids)
                accuracy = (output_values.argmax(1) == celltype_labels).sum().item()
                total_error += (1 - accuracy / len(input_gene_ids)) * len(input_gene_ids)
                total_num += len(input_gene_ids)

            preds = output_values.argmax(1).cpu().numpy()
            predictions.append(preds)
            probs = F.softmax(output_values, dim=-1).cpu().numpy()
            confidences.append(probs)

    if not is_inference_task:
        wandb.log(
            {
                "valid/mse": total_loss / total_num,
                "valid/err": total_error / total_num,
                "epoch": epoch,
            },
        )

    if return_raw:
        return np.concatenate(predictions, axis=0), np.concatenate(confidences, axis=0)

    return total_loss / total_num, total_error / total_num


def test(
    model: nn.Module,
    loader: DataLoader,
    adata: AnnData,
    true_cell_type_ids: List[int] | None,
    ref_id2type: dict,
    pad_vocab,
    criterion_cls,
    config,
    epoch,
    logger,
    device
):
    model.eval()
    predictions, confidences = evaluate(
        model,
        loader=loader,
        pad_vocab=pad_vocab,
        criterion_cls=criterion_cls,
        config=config,
        epoch=epoch,
        device=device,
        return_raw=True
    )
    results = None
    accuracy_by_cell_type = None
    wrong_predictions = None
    if true_cell_type_ids is not None:
        unique_true_cell_type_ids = [str(i) for i in set(true_cell_type_ids)]
        accuracy = accuracy_score(true_cell_type_ids, predictions)
        unique_true_cell_type_ids = list(map(int, unique_true_cell_type_ids))
        accuracy_by_cell_type = {}
        for cell_type in unique_true_cell_type_ids:
            indices_of_cell_type = [i for i, label in enumerate(true_cell_type_ids) if label == cell_type]
            if indices_of_cell_type:
                true_labels_for_cell_type = [true_cell_type_ids[i] for i in indices_of_cell_type]
                predictions_for_cell_type = [predictions[i] for i in indices_of_cell_type]
                accuracy_for_cell_type = accuracy_score(true_labels_for_cell_type, predictions_for_cell_type)
                accuracy_by_cell_type[ref_id2type[str(cell_type)]] = accuracy_for_cell_type

        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            true_cell_type_ids, predictions, zero_division=0
        )
        total_support = np.sum(support_per_class)
        weighted_precision = np.sum(precision_per_class * support_per_class) / total_support
        weighted_recall = np.sum(recall_per_class * support_per_class) / total_support
        weighted_f1 = np.sum(f1_per_class * support_per_class) / total_support
        kappa_coefficient = cohen_kappa_score(true_cell_type_ids, predictions, weights='quadratic')

        wrong_predictions = {}
        for gt, pred, idx in zip(true_cell_type_ids, predictions, range(len(predictions))):
            if gt != pred:
                wrong_predictions[adata.obs.index[idx]] = [ref_id2type[str(gt)], ref_id2type[str(pred)]]

        print('=' * 30)
        print(f'<Evaluation Statistics>')
        print(
            f"Accuracy: {accuracy:.3f}\n"
            f"Precision: {weighted_precision:.3f}\n"
            f"Recall: {weighted_recall:.3f}\n"
            f"Macro F1: {weighted_f1:.3f}\n"
            f"Kappa Coefficient: {kappa_coefficient:.3f}"
        )
        print('=' * 30)

        results = {
            "test/accuracy": accuracy,
            "test/precision": weighted_precision,
            "test/recall": weighted_recall,
            "test/macro_f1": weighted_f1,
            "test/kappa_coefficient": kappa_coefficient
        }

    return predictions, results, accuracy_by_cell_type, wrong_predictions, confidences


#%% END
