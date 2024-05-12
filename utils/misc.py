#%%
from anndata import AnnData

from .protocol_prelude import *


#%%
class ProtocolWandB:
    def __init__(self, configs):
        self.configs = configs
        self.wandb_configs = configs['wandb_configs']
        os.environ["WANDB_MODE"] = self.wandb_configs['mode']

    def create_wandb_project(self):
        return wandb.init(
            config=self.configs,
            project=self.wandb_configs['project'],
            reinit=self.wandb_configs['reinit'],
            settings=wandb.Settings(start_method="fork"),
            name=self.wandb_configs['name']
        )


#%%
class Preprocessor:
    def __init__(
        self,
        logger,
        use_key: Optional[str] = None,
        filter_gene_by_counts: Union[int, bool] = False,
        filter_cell_by_counts: Union[int, bool] = False,
        normalize_total: Union[float, bool] = 1e4,
        log1p: bool = False,
        subset_hvg: Union[int, bool] = False,
        hvg_use_key: Optional[str] = None,
        hvg_flavor: str = "seurat_v3",
    ):
        """
        use_key (:class:`str`, optional):
            The key of :class:`~anndata.AnnData` to use for preprocessing.
        filter_gene_by_counts (:class:`int` or :class:`bool`, default: ``False``):
            Whther to filter genes by counts, if :class:`int`, filter genes with counts
        filter_cell_by_counts (:class:`int` or :class:`bool`, default: ``False``):
            Whther to filter cells by counts, if :class:`int`, filter cells with counts
        normalize_total (:class:`float` or :class:`bool`, default: ``1e4``):
            Whether to normalize the total counts of each cell to a specific value.
        log1p (:class:`bool`, default: ``True``):
            Whether to apply log1p transform to the normalized data.
        subset_hvg (:class:`int` or :class:`bool`, default: ``False``):
            Whether to subset highly variable genes.
        hvg_use_key (:class:`str`, optional):
            The key of :class:`~anndata.AnnData` to use for calculating highly variable
            genes. If :class:`None`, will use :attr:`adata.X`.
        hvg_flavor (:class:`str`, default: ``"seurat_v3"``):
            The flavor of highly variable genes selection. See
            :func:`scanpy.pp.highly_variable_genes` for more details.
        """
        self.use_key = use_key
        self.filter_gene_by_counts = filter_gene_by_counts
        self.filter_cell_by_counts = filter_cell_by_counts
        self.normalize_total = normalize_total
        self.log1p = log1p
        self.subset_hvg = subset_hvg
        self.hvg_use_key = hvg_use_key
        self.hvg_flavor = hvg_flavor
        self.logger = logger

    def __call__(self, adata, batch_key: Optional[str] = None) -> Dict:
        key_to_process = self.use_key
        if key_to_process == "X":
            key_to_process = None  # the following scanpy apis use arg None to use X
        is_logged = self.check_logged(adata, obs_key=key_to_process)

        # filter genes
        if self.filter_gene_by_counts:
            self.logger.info("Filtering genes by counts ...")
            sc.pp.filter_genes(
                adata,
                min_counts=self.filter_gene_by_counts
                if isinstance(self.filter_gene_by_counts, int)
                else None,
            )

        # filter cells
        if self.filter_cell_by_counts > 0:
            self.logger.info("Filtering cells by counts ...")
            sc.pp.filter_cells(
                adata,
                min_counts=self.filter_cell_by_counts
                if isinstance(self.filter_cell_by_counts, int)
                else None,
            )

        # normalize total
        if self.normalize_total:
            self.logger.info("Normalizing total counts ...")
            sc.pp.normalize_total(
                adata,
                target_sum=self.normalize_total
                if isinstance(self.normalize_total, float)
                else None
            )

        # log1p
        if self.log1p:
            self.logger.info("Log1p transforming ...")
            if is_logged:
                self.logger.warning(
                    "The input data seems to be already log1p transformed. "
                    "Set `log1p=False` to avoid double log1p transform."
                )
                self.logger.warning(
                    "Skip log1p transformation ..."
                )
            else:
                sc.pp.log1p(adata)

        # extract hvg
        if self.subset_hvg:
            self.logger.info("Subsetting highly variable genes ...")
            if batch_key is None:
                self.logger.warning(
                    "No batch_key is provided, will use all cells for HVG selection."
                )
            sc.pp.highly_variable_genes(
                adata,
                layer=self.hvg_use_key,
                n_top_genes=self.subset_hvg
                if isinstance(self.subset_hvg, int)
                else None,
                batch_key=batch_key,
                flavor=self.hvg_flavor,
                subset=True
            )

    def check_logged(self, adata, obs_key: Optional[str] = None) -> bool:
        """
        Check if the data is already log1p transformed.

        Args:

        adata (:class:`AnnData`):
            The :class:`AnnData` object to preprocess.
        obs_key (:class:`str`, optional):
            The key of :class:`AnnData.obs` to use for batch information. This arg
            is used in the highly variable gene selection step.
        """
        data = adata.X
        max_, min_ = data.max(), data.min()
        if max_ > 30:
            return False
        if min_ < 0:
            return False
        non_zero_min = data[data > 0].min()
        if non_zero_min >= 1:
            return False
        return True


#%% Save file
def save_h5ad(adata: AnnData, save_dir: Path, compression_opts: int = 6) -> str:
    data_dir = save_dir / 'input_data.h5ad'
    adata.write(data_dir, compression='gzip', compression_opts=compression_opts)
    return str(data_dir)


#%% Data binning
def _digitize(x: np.ndarray, bins: np.ndarray, side="both") -> np.ndarray:
    """
    Digitize the data into bins. This method spreads data uniformly when bins
    have same values.

    Args:

    x (:class:`np.ndarray`):
        The data to digitize.
    bins (:class:`np.ndarray`):
        The bins to use for digitization, in increasing order.
    side (:class:`str`, optional):
        The side to use for digitization. If "one", the left side is used. If
        "both", the left and right side are used. Default to "one".

    Returns:

    :class:`np.ndarray`:
        The digitized data.
    """
    assert x.ndim == 1 and bins.ndim == 1

    left_digits = np.digitize(x, bins)
    if side == "one":
        return left_digits

    right_difits = np.digitize(x, bins, right=True)

    rands = np.random.rand(len(x))  # uniform random numbers

    digits = rands * (right_difits - left_digits) + left_digits
    digits = np.ceil(digits).astype(np.int64)
    return digits


def binning(
    row: Union[np.ndarray, torch.Tensor], n_bins: int, logger
) -> Union[np.ndarray, torch.Tensor]:
    """Binning the row into n_bins."""
    dtype = row.dtype
    return_np = False if isinstance(row, torch.Tensor) else True
    row = row.cpu().numpy() if isinstance(row, torch.Tensor) else row

    if row.max() == 0:
        logger.warning(
            "The input data contains row of zeros. Please make sure this is expected."
        )
        return (
            np.zeros_like(row, dtype=dtype)
            if return_np
            else torch.zeros_like(row, dtype=dtype)
        )

    if row.min() <= 0:
        non_zero_ids = row.nonzero()
        non_zero_row = row[non_zero_ids]
        bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
        non_zero_digits = _digitize(non_zero_row, bins)
        binned_row = np.zeros_like(row, dtype=np.int64)
        binned_row[non_zero_ids] = non_zero_digits
    else:
        bins = np.quantile(row, np.linspace(0, 1, n_bins - 1))
        binned_row = _digitize(row, bins)
    return torch.from_numpy(binned_row) if not return_np else binned_row.astype(dtype)


#%% END
