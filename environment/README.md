# environment

| File | Contents |
|---|---|
| `requirements-modeling.txt` | Pinned packages for everything that runs on a CPU: modelling, statistics, tables and figures (`04_modeling/`, `07_baselines/scripts/evaluate.py` and onwards, `08_external_validation/` steps 4–6). Python 3.11. |
| `modeling_env_pip_freeze.txt` | Full `pip freeze` of the modelling environment used for the manuscript (72-core CPU node). |
| `gn1_evo2_conda_packages.json` | Package list of the Evo 2 environment used for GPU feature extraction. |
| `gn1_hardware_software_snapshot.txt` | CPU, GPU, driver, CUDA and OS of the GPU feature-extraction node (8× RTX 5880 Ada / RTX A6000). |

```bash
conda create -n merge_ml -c conda-forge python=3.11
conda activate merge_ml
pip install -r environment/requirements-modeling.txt
```

Use the `conda-forge` channel: a Python 3.11 from `defaults` has shipped a `pyexpat` with
a mismatched ABI (`undefined symbol: XML_SetAllocTrackerActivationThreshold`) that makes
pip fail on start-up.

The committed results were produced with these versions on one 16-core node (12 cores,
8 threads per library), all four tasks in one run. XGBoost, CatBoost and the tree
ensembles use all available cores (`n_jobs=-1`), and their floating-point summation order
depends on the thread count, so on other hardware a few fold-level metrics move in the
fourth decimal: run on the same tables, a 72-core node and this node gave the same
task-level MERGE numbers in Tables 2, 4 and 6 except one 1:1 AUROC, which differed by
4e-5, and the same panels and algorithms for the deployed models. Re-fitted tree models
can differ slightly in their predictions, so the models used by the web server and
`merge-cli` are the committed files in `05_results/trained_models/`. Other numpy/scipy builds change
out-of-fold scores in the last floating-point digit (about 1e-16), which leaves every
metric unchanged but changes file checksums.

## GPU feature extraction

Feature extraction used several mutually incompatible environments; each scoring script
in `02_feature_extraction/shared/` documents its own requirements.

* **Evo 2** — `evo2` 0.5.3 with Transformer Engine. FP8 inference needs compute capability
  8.9 or higher (RTX 5880 Ada); `02_feature_extraction/shared/evo2_compatibility_patches/`
  is the BF16 fallback shim for GPUs without FP8. Set
  `NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE=1` to load the official weights.
* **Nucleotide Transformer, NT-v2, HyenaDNA, GENERATOR, Carbon-3B** — PyTorch 2.7 with
  transformers **4.57.6** (5.x breaks the NT-v2 remote code). Language models run in fp32;
  Carbon-3B in bf16.
* **Enformer** — TensorFlow 2 with `tensorflow-hub` and `kipoiseq` (indel and MNV sequence
  extraction). TensorFlow GPU kernels are not bit-deterministic, so repeated Enformer runs
  differ slightly.
* **AlphaGenome** — no local checkpoint: the official `alphagenome` client
  (`alphagenome.models.dna_client`) with your own API key.
