# ranktwosample

Rank-based two-sample testing code for permutation-calibrated multivariate goodness-of-fit and power experiments.

This repository implements a **non-symmetric rank $\chi^2$ / G-test** and a **subspace-sliced extension**, together with dataset generators and several baseline methods commonly used in two-sample testing benchmarks.

## What is included

The repository is organized as follows:

```text
src/
├── ranktwosample/
│   ├── rank_two_sample.py
│   ├── rank_two_sample_subspaces.py
│   └── power_datasets.py
├── baselines/
│   ├── c2st.py
│   ├── hotelling_t2.py
│   ├── kernel_mmd.py
│   ├── rank_chi2.py
│   ├── rank_chi2_subspaces.py
│   ├── sliced_ot.py
│   └── tuned_mmd.py
└── scripts/
    ├── run_power_bench.py
    ├── power_rank_subspaces.py
    └── cifar10c_power_rank_subspaces.py
````

## Main components

### 1. Core test: `rank_two_sample.py`

Implements the main **non-symmetric rank (\chi^2) / G-test** with:

* permutation calibration
* optional CuPy GPU backend
* `chi2` and `gtest` statistics
* early stopping for permutations
* two reference modes:

  * `shared_pool`: practical pooled-reference version
  * `fresh_iid`: fresh-reference version aligned with the theoretical construction

### 2. Subspace extension: `rank_two_sample_subspaces.py`

Implements a **subspace-sliced** version that aggregates rank-based statistics across multiple random or user-specified low-dimensional subspaces.

Useful when the ambient dimension is too large for a direct full-dimensional discretization.

### 3. Dataset generators: `power_datasets.py`

Provides synthetic data generators for power experiments, including:

* location shift under correlated Gaussian models
* scale / shape alternatives
* multimodal mixture alternatives
* dependence changes via copula-based constructions

### 4. Baselines

The `src/baselines/` folder contains optional baselines for comparison, including:

* classifier two-sample test (C2ST)
* Hotelling's (T^2)
* kernel MMD
* tuned MMD
* sliced Wasserstein / sliced OT
* rank-based baselines and subspace variants

### 5. Experiment scripts

The `src/scripts/` folder contains ready-to-run scripts for:

* synthetic power and calibration benchmarks
* subspace-rank experiments
* CIFAR-10 vs CIFAR-10-C corruption experiments

---

## Setup

This repository currently follows a simple **research-code layout**. The easiest way to run it is by keeping `src/` on your `PYTHONPATH`.

### Clone the repository

```bash
git clone https://github.com/josemanuel22/ranktwosample.git
cd ranktwosample
```

### Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install likely dependencies

At minimum, you will probably want:

```bash
pip install numpy scipy scikit-learn
```

Optional:

```bash
pip install cupy-cuda12x
```

Adjust the CuPy package to your local CUDA version if needed.

### Run code from the repository root

```bash
export PYTHONPATH=src
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
```

---

## Quick start

## Run a synthetic benchmark

The main benchmark runner supports multiple methods, multiple sample sizes, and both **power** and **calibration** experiments.

Example:

```bash
PYTHONPATH=src python src/scripts/run_power_bench.py \
  --experiment power \
  --methods rank_chi2 rank_chi2_subspaces mmd c2st hotelling swd \
  --Ns 100 200 500 \
  --Ms 100 200 500 \
  --K_ref 64 \
  --B 300
```

This script is designed to output CSV summaries with rejection rates and standard errors.

---

## Run the subspace rank test script

Example:

```bash
PYTHONPATH=src python src/scripts/power_rank_subspaces.py \
  --scenario shift \
  --d 8 \
  --N 600 \
  --K 600 \
  --delta 0.5 \
  --L 64 \
  --k-dim 2 \
  --K-ref 64 \
  --B 500 \
  --R 200 \
  --engine auto
```

Supported scenario families in this script include examples such as:

* `shift`
* `scale`
* `corr`
* `tdf`
* `mixture`

---

## CIFAR-10 / CIFAR-10-C benchmark

There is also a script for corruption-shift experiments on CIFAR-10 vs CIFAR-10-C.

Example layout expected by the script:

```text
data/
├── cifar-10-batches-py/
│   └── test_batch
└── gaussian_noise.npy
   jpeg_compression.npy
   ...
```

Example invocation:

```bash
PYTHONPATH=src python src/scripts/cifar10c_power_rank_subspaces.py \
  --cifar-root ./data/cifar-10-batches-py \
  --c10c-root ./data
```

---

## Notes on the methodology

The main rank-based procedure is **non-symmetric**: one sample is treated as the target sample and the other as a reference / calibration pool.

The implementation supports both:

* a practical pooled-reference mode (`shared_pool`)
* a fresh-reference empirical mode (`fresh_iid`)

The subspace extension aggregates low-dimensional contributions across multiple slices, which can improve scalability in higher dimensions.

---

## Typical use cases

This repository is useful for:

* evaluating two-sample test power under controlled synthetic alternatives
* comparing rank-based tests against MMD, sliced Wasserstein, Hotelling, and C2ST
* studying high-dimensional testing via random low-dimensional subspaces
* running CIFAR-10-C corruption benchmarks

---

## Reproducibility tips

* Fix random seeds whenever possible
* keep `B` (number of permutations) large enough for stable p-values
* report both rejection rates and uncertainty across repetitions
* for higher-dimensional settings, prefer the subspace version over the full discretization

---

## Caveats

This repository is currently organized primarily as a **research codebase**, not yet as a fully packaged Python library. In particular:

* installation is easiest via `PYTHONPATH=src`
* some baselines are optional and may be skipped if dependencies are missing
* GPU support is optional and relies on CuPy

---
## License

This repository's source code is released under the MIT License. See the `LICENSE` file.

## Third-party assets

- CIFAR-10: official dataset page: https://www.cs.toronto.edu/~kriz/cifar.html
- CIFAR-10-C: CC BY 4.0, https://doi.org/10.5281/zenodo.2535967
- hendrycks/robustness code repository: Apache-2.0
