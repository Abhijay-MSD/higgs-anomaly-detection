# A jet-multiplicity confound in reconstruction-based anomaly detection

Code and results for the paper "A Jet-Multiplicity Confound in Reconstruction-Based Anomaly Detection: Evidence from a Higgs-to-Tau-Pair Proxy Signal" by Abhijay Arun (version v2-revision; earlier title: "Evaluating Representation Learning for Model-Independent Anomaly Detection in High-Energy Particle Collisions").

We train PCA, an autoencoder (AE) and a variational autoencoder (VAE) on **simulated background events only**, then test how well their reconstruction error flags an unseen **proxy signal** (the Higgs events in the dataset) and controlled synthetic shifts. We also analyse what the latent codes contain. The Higgs events are a known process used as a stand-in. Nothing here claims to find new physics.

## Main findings

- **Jet count is a confound.** Signal and background have different jet-count mixes, and raw reconstruction error partly tracks jet count. Jet count alone, used as a score, reaches a raw AUC of 0.587, the same range as the reconstruction scores (0.547 to 0.615). Ranking each score within its jet-count group removes this.
- **After that correction, reconstruction error is a weak detector.** Only PCA with 2 or 4 components stays above chance; the larger edge, at 4 components, is AUC 0.553 (95% bootstrap interval 0.550 to 0.555) and 1.6% signal efficiency at 1% false positive rate (chance is 1.0%), and it is reproduced on the held-out `sig_clf` half of the signal (0.552, 1.59%). The AE and VAE are at or slightly below chance (mean AUC 0.47 to 0.50 across latent sizes).
- **Further checks leave this unchanged.** Isolation-forest and nearest-neighbour baselines, Mahalanobis and Gaussian-mixture scores on the latent codes, the VAE's negative ELBO, a score masked to defined features, and retraining without the jet-count inputs all give corrected results near chance.
- **The signal is separable.** A supervised reference MLP reaches AUC 0.911 and 33.7% efficiency at 1% false positive rate. It is a reference only and is never used for anomaly detection.
- **Independent synthetic shifts are detected well, correlated ones are not.** With all 18 shiftable features moved by 2 standard deviations, the AE (latent size 8) catches about 100% of shifted events at 1% false alarms, against 57% for PCA (latent size 8). Scaling all energy-like features together by up to 20% is almost invisible to every detector (AUC 0.488 to 0.525).
- **Latent codes.** Jet count can be read from the codes with 94% to 100% accuracy (guessing gives 45%). A linear probe finds the proxy signal in the codes (AUC within jet count 0.75 to 0.76 at latent size 16, against 0.80 on the raw inputs). The VAE uses at most about 7 latent dimensions.

## Repository layout

- `src/` all code (preprocessing, models, evaluation, experiments, figures)
- `results/` result tables: `results.csv` (main grid), `stratified.csv`, `shift.csv`, `latent_summary.csv`, `latent_features.csv`, `bootstrap_pct.csv`
- `results/revision/` tables for the further checks added in peer review (files A to J, named after their check)
- `figures/` paper figures (PNG and PDF)
- `docs/` project notes

Model weights and the dataset are not included.

## Setup

Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Download `atlas-higgs-challenge-2014-v2.csv.gz` from the CERN Open Data Portal (record 328, https://opendata.cern.ch/record/328) and place it in `data/raw/`. The data are released under the Creative Commons Zero v1.0 Universal license.

## Reproduce

```bash
python src/preprocess.py             # clean, split and scale the data
caffeinate -i python src/run_grid.py # PCA, AE and VAE grid, about 90 minutes on a laptop
python src/supervised_bound.py       # supervised reference model
python src/stratified_eval.py        # jet-count-corrected evaluation
python src/shift_study.py            # synthetic shift study
python src/latent_analysis.py        # linear probes and latent samples
python src/bootstrap_pca.py          # bootstrap intervals
python src/figures.py                # figures (needs umap-learn)
```

Further checks added in peer review (outputs go to `results/revision/`):

```bash
python src/revision_analyses.py      # sig_clf confirmation, correlations, raw efficiencies, baselines, latent-code scores, centrality, masked score
python src/baseline_knn.py           # nearest-neighbour baseline (appends to D_baselines.csv; run after revision_analyses.py)
python src/retrain_nojet.py          # retrain without PRI_jet_num and the two jet indicators (about 1 hour)
python src/energy_shift.py           # correlated energy-scale shifts
python src/diag_jet_masked.py        # masked scores by jet count, and the features where signal is rebuilt better
python src/make_revision_tables.py   # LaTeX tables for the paper appendix (results/revision/tables.tex)
python src/paper_numbers.py          # prints the derived numbers quoted in the paper
```

`caffeinate` keeps a Mac awake and is macOS only, so drop it elsewhere. Splits and models use fixed seeds (split seed 0, model seeds 0 to 2), and reruns on our machine gave identical results.

Signal events are never used to train, validate, early-stop, scale or tune any model. They are used only for evaluation, and for the labelled reference probes.

## Limitations

- 9 of the 24 neural runs reached the 300-epoch cap before early stopping.
- The jet-count issue was found after seeing the first results, so the jet-stratified evaluation is post hoc (it is confirmed on `sig_clf`, which played no part in choosing it).
- PCA is one deterministic fit, and the bootstrap intervals cover test-sample noise only.
- The synthetic shifts move features independently, which is easier to detect than a realistic signal.
- One simulated dataset, and the signal is a proxy that shares its final state with the Z to tau-pair background. The public data do not record which process produced each background event, so no background process could be held out as a second proxy.

## Citation

Repository: https://github.com/Abhijay-MSD/higgs-anomaly-detection

Archived release (DOI): https://doi.org/10.5281/zenodo.23325916

Please cite this software as:

> Arun, A. (2026). A Jet-Multiplicity Confound in Reconstruction-Based Anomaly Detection: Evidence from a Higgs-to-Tau-Pair Proxy Signal (v2-revision) [Software]. Zenodo. https://doi.org/10.5281/zenodo.23325916

and cite the dataset:

> ATLAS collaboration (2014). Dataset from the ATLAS Higgs Boson Machine Learning Challenge 2014. CERN Open Data Portal. DOI:10.7483/OPENDATA.ATLAS.ZBP2.M5T8

Neither ATLAS nor CERN endorses works produced using these data.

## AI assistance

Claude (Anthropic) was used to help write and debug the code, including the scripts for the further checks, to run some of those checks, and to help draft and edit the paper. The author ran the main experiments, reviewed all code and results, and takes full responsibility for the content.

## License

Code: MIT (see `LICENSE`). The dataset is CC0 and is not redistributed here.