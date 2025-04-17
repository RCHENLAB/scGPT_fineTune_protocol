# %%
from .protocol_prelude import *
from .misc import binning


def tokenize_batch(
    data: np.ndarray,
    gene_ids: np.ndarray,
    return_pt: bool = True,
    append_cls: bool = True,
    include_zero_gene: bool = False,
    cls_id: int = "<cls>"
) -> List[Tuple[Union[torch.Tensor, np.ndarray]]]:
    if data.shape[1] != len(gene_ids):
        raise ValueError(
            f"Number of features in data ({data.shape[1]}) does not match "
            f"number of gene_ids ({len(gene_ids)})."
        )
    tokenized_data = []

    for i in range(len(data)):
        row = data[i]
        mod_types = None
        if include_zero_gene:
            values = row
            genes = gene_ids
        else:
            idx = np.nonzero(row)[0]
            values = row[idx]
            genes = gene_ids[idx]
        if append_cls:
            genes = np.insert(genes, 0, cls_id)
            values = np.insert(values, 0, 0)
        if return_pt:
            genes = torch.from_numpy(genes).long()
            values = torch.from_numpy(values).float()
        tokenized_data.append((genes, values))
    return tokenized_data


def pad_batch(
    batch: List[Tuple],
    max_len: int,
    vocab: Vocab,
    pad_token: str = "<pad>",
    pad_value: int = 0,
    cls_appended: bool = True
) -> Dict[str, torch.Tensor]:
    max_ori_len = max(len(batch[i][0]) for i in range(len(batch)))
    max_len = min(max_ori_len, max_len)
    pad_id = vocab[pad_token]
    gene_ids_list = []
    values_list = []
    for i in range(len(batch)):
        gene_ids, values = batch[i]
        if len(gene_ids) > max_len:
            # sample max_len genes
            if not cls_appended:
                idx = np.random.choice(len(gene_ids), max_len, replace=False)
            else:
                idx = np.random.choice(len(gene_ids) - 1, max_len - 1, replace=False)
                idx = idx + 1
                idx = np.insert(idx, 0, 0)
            gene_ids = gene_ids[idx]
            values = values[idx]
        if len(gene_ids) < max_len:
            gene_ids = torch.cat(
                [
                    gene_ids,
                    torch.full(
                        (max_len - len(gene_ids),), pad_id, dtype=gene_ids.dtype
                    ),
                ]
            )
            values = torch.cat(
                [
                    values,
                    torch.full((max_len - len(values),), pad_value, dtype=values.dtype),
                ]
            )
        gene_ids_list.append(gene_ids)
        values_list.append(values)
    batch_padded = {
        "genes": torch.stack(gene_ids_list, dim=0),
        "values": torch.stack(values_list, dim=0),
    }

    return batch_padded


def tokenize_and_pad_batch(
    data: np.ndarray,
    gene_ids: np.ndarray,
    max_len: int,
    vocab: Vocab,
    pad_token: str,
    pad_value: int,
    append_cls: bool = True,
    include_zero_gene: bool = False,
    cls_token: str = "<cls>",
    return_pt: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Tokenize and pad a batch of data. Returns a dictionary of tensors.
    """
    cls_id = vocab[cls_token]
    tokenized_data = tokenize_batch(
        data,
        gene_ids,
        return_pt=return_pt,
        append_cls=append_cls,
        include_zero_gene=include_zero_gene,
        cls_id=cls_id,
    )

    batch_padded = pad_batch(
        tokenized_data,
        max_len,
        vocab,
        pad_token,
        pad_value,
        cls_appended=append_cls
    )
    return batch_padded


def prepare_data(
    tokenized_and_padded_batch: Dict[str, torch.Tensor],
    examples
) -> List[Dict[str, torch.Tensor]]:
    prepared_examples = []
    for i, example in enumerate(examples):
        cell_type_id = example['cell_type_id']
        genes = tokenized_and_padded_batch['genes'][i]
        expressions = tokenized_and_padded_batch['values'][i]
        prepared_examples.append({
            'cell_type_id': cell_type_id,
            'genes': genes,
            'expressions': expressions
        })

    return prepared_examples


@dataclass
class DataCollator:
    do_padding: bool = True
    pad_token_id: Optional[int] = None
    pad_value: int = 0
    do_mlm: bool = True
    do_binning: bool = True
    mlm_probability: float = 0.15
    mask_value: int = -1
    max_length: Optional[int] = None
    sampling: bool = True
    reserve_keys: List[str] = field(default_factory=lambda: [])
    keep_first_n_tokens: int = 1
    data_style: str = "cls"
    filtered_gene_list: List[int] = None
    vocab: Vocab = None
    append_cls: bool = True,
    include_zero_gene: bool = False,
    pad_token: str = "<pad>"
    cls_token: str = "<cls>"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self):
        if self.do_padding:
            if self.pad_token_id is None:
                raise ValueError("`pad_token_id` is required if `do_padding`.")
            if self.max_length is None:
                raise ValueError("`max_length` is required if `do_padding`.")

        if isinstance(self.mlm_probability, float):
            if self.mlm_probability <= 0 or self.mlm_probability >= 1:
                raise ValueError("`mlm_probability` must be between 0 and 1.")
        elif isinstance(self.mlm_probability, (list, tuple)):
            if min(self.mlm_probability) <= 0 or max(self.mlm_probability) >= 1:
                raise ValueError("`mlm_probability` must be between 0 and 1.")
        else:
            raise ValueError("`mlm_probability` must be a float or iterable of floats.")

        if isinstance(self.reserve_keys, str):
            self.reserve_keys = [self.reserve_keys]

        if self.keep_first_n_tokens < 0 or self.keep_first_n_tokens > self.max_length:
            raise ValueError(
                "`keep_first_n_tokens` must be between 0 and `max_length` "
                f"({self.max_length})."
            )

        if self.data_style not in ["pcpt", "gen", "both", 'cls']:
            raise ValueError("`data_style` must be one of 'pcpt', 'gen', 'both', 'cls'.")

    def __call__(
            self, examples: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        if len(self.reserve_keys) > 0:
            assert all(key in examples[0] for key in self.reserve_keys), (
                f"reserve_keys must be a subset of the keys in the examples. "
                f"Got {self.reserve_keys} but expected keys in {list(examples[0].keys())}."
            )
        data = np.array([example['data'] for example in examples])
        gene_ids = np.array(self.filtered_gene_list)

        # Step 1: binning data
        if self.do_binning:
            for i, expressions in enumerate(data):
                expressions[self.keep_first_n_tokens:] = binning(
                    row=expressions[self.keep_first_n_tokens:],
                    n_bins=51,
                )
                data[i] = expressions

        # Step 2: Tokenize and Pad Batch
        tokenized_and_padded_batch = tokenize_and_pad_batch(
            data,
            gene_ids,
            self.max_length,
            self.vocab,
            self.pad_token,
            self.pad_value,
            append_cls=self.append_cls,
            include_zero_gene=self.include_zero_gene,
            cls_token=self.cls_token,
            return_pt=True
        )

        # Step 3: process data into correct format
        prepared_examples = prepare_data(tokenized_and_padded_batch, examples)

        # Step 4: Create train-ready data dict
        data_dict = self._call_cls(prepared_examples)

        # add reserved keys
        for key in self.reserve_keys:
            data_ = [example[key] for example in prepared_examples]
            data_dict[key] = torch.stack(data_, dim=0)

        return data_dict

    def _call_cls(
            self,
            examples: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        call if training object is Cell Annotation (Classifier)
        e.g. examples -> format:
       [
            {
                'cell_type_id': 1  # int8
                'genes': [1, 3, 42, 53, ..., 60603]  # List[int]
                'expressions': [0.11, 2., ..., 0.3]  # List[float16]
            },
            ...
        ]
        :param examples: List of Dict with keys 'cell_type_id', 'genes', 'expressions'
        :return: data_dict: Dictionary with keys 'input_gene_ids', 'input_expr', 'target_expr', 'cell_types'
        """

        input_gene_ids = []
        target_expressions = []
        cell_types = []
        for i in range(len(examples)):
            genes = torch.tensor(examples[i]["genes"], dtype=torch.int32)
            expressions = torch.tensor(examples[i]["expressions"], dtype=torch.float16)
            input_gene_ids.append(genes)
            target_expressions.append(expressions)
            cell_types.append(examples[i]["cell_type_id"])

        input_gene_ids = torch.stack(input_gene_ids, dim=0)
        target_expressions = torch.stack(target_expressions, dim=0)

        # mask
        if self.do_mlm:
            input_expressions = self._mask(
                target_expressions, self.keep_first_n_tokens
            )
        else:
            input_expressions = target_expressions

        return {
            "input_gene_ids": input_gene_ids,
            "input_expr": input_expressions,
            "target_expr": target_expressions,
            "cell_types": torch.tensor(cell_types, dtype=torch.long)
        }

    def get_mlm_probability(self) -> float:
        if isinstance(self.mlm_probability, float):
            return self.mlm_probability
        elif isinstance(self.mlm_probability, list):
            # random choose a probability
            return np.random.choice(self.mlm_probability)
        else:
            raise ValueError(
                "mlm_probability must be a float or a list of floats, "
                f"but got {self.mlm_probability}."
            )

    def _mask(
            self, expressions: torch.Tensor, keep_first_n_tokens: int = 0
    ) -> torch.Tensor:
        if keep_first_n_tokens > 0:
            result_ = self._mask(
                expressions[:, keep_first_n_tokens:],
                keep_first_n_tokens=0,
            )
            return torch.cat([expressions[:, :keep_first_n_tokens], result_], dim=1)
        shape = expressions.shape

        probability_matrix = torch.full(shape, self.get_mlm_probability())
        # set padded postion probability to 0
        probability_matrix[expressions.eq(self.pad_value)] = 0
        if self.keep_first_n_tokens > 0:
            probability_matrix[:, : self.keep_first_n_tokens] = 0

        mask = torch.bernoulli(probability_matrix).bool()

        masked_expressions = expressions.masked_fill(mask, self.mask_value)
        return masked_expressions
