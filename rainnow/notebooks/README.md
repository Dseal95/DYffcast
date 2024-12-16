# 📚 Rainnow Notebook Glossary
This glossary provides an overview of the **notebooks** in this project. They are organised into **five** main sections:

| Section | Description | Directory 📁 |
|---------|-------------|-----------|
| [ConvLSTM](#convlstm) | ConvLSTM model implementation and evaluation | `conv_lstm/` |
| [DYffusion](#dyffusion) | DYffcast Interpolator I(φ) and Forecastor F(θ) networks | `dyffusion/` |
| [IMERG Data](#imerg-data) | IMERG dataset processing and analysis | `imerg_data/` |
| [STEPS](#pysteps_steps) |  PySteps implementation of STEPS nowcast | `pysteps_steps/` |
| [Research Plots](#research_plots) | Plots and visualisations used throughout/in the research | `research_plots/` |

## ConvLSTM 🔗
Notebooks related to the ConvLSTM model implementation, training, and evaluation.

| Notebook | Description |
|----------|-------------|
| `convlstm_evaluation.ipynb` | Evaluation of ConvLSTM model(s) on the test dataset. Calculates entire sequence and per timestep ($/t$) metrics including MSE, LPIPS, and CSI. |
| `convlstm_exploration.ipynb` | Exploration and training of ConvLSTM on the IMERG dataset. |
| `convlstm_inference_one_shot.ipynb` | One-shot inference using a trained ConvLSTM model. |
| `convlstm_inference_roll_out.ipynb` | Roll-out inference using a trained ConvLSTM model. |

## DYffusion 🔮
Notebooks related to DYffcast (DYffusion) Interpolator I(φ) and Forecastor F(θ) networks.

| Notebook | Description |
|----------|-------------|
| `forecastor_model_evaluation.ipynb` | Evaluation of DYffusion Forecastor F(θ) on the test dataset. Calculates entire sequence and per timestep ($/t$) metrics including **MSE**, **CRPS**, **SSR**, **LPIPS**, and **CSI**. |
| `forecastor_model_inference.ipynb` | Sampling from a trained DYffusion model. |
| `interpolator_deterministic_evaluation.ipynb` | Deterministic Interpolator I(φ) inference and experiments for assessing different loss functions. |
| `interpolator_stochastic_evaluation.ipynb` | Interpolator I(φ) stochasticity experiments. Sweeping over different dropout rates and calculating metrics on the test dataset. |
| `interpolator_stochastic_inference.ipynb` | Sampling and plotting interpolated values from trained Interpolators, I(φ). This notebook also plots the ensemble plots and the member distributions. |

## IMERG Data 🗺️
Notebooks for processing, analysing, and visualising IMERG data.

| Notebook | Description |
|----------|-------------|
| `imerg_cb_loss_weight_matrix.ipynb` | Illustration of the class weights used in the **CBLoss()** (combined MAE and MSE). |
| `imerg_data_distribution_comparison.ipynb` | Comparison of data distributions: IMERG, Sea Surface Temperatures (SST), and Navier Stokes (NS). |
| `imerg_data_distributions_percentiles.ipynb` | Exploration of data normalisation techniques (for rainfall). |
| `imerg_data_download.ipynb` | Downloading IMERG data from NASA PPS server using `gpm_api` library. |
| `imerg_data_exploration.ipynb` | Exploration of raw IMERG data (and patching into H x W tiles). |
| `imerg_patchify.ipynb` | Exploration of patching IMERG data into H x W tiles and different patching methods. |
| `imerg_rainfall_classes.ipynb` | Analysis of pixel percentages for discrete rainfall classifications in the "train", "val", and "test" datasets. |
| `imerg_sequences.ipynb` | Sequencing and plotting IMERG data. |

## Research Plots 📈
Notebooks containing plots and visualisations used throughout/in research.

| Notebook | Description |
|----------|-------------|
| `eval_metrics/` | Directory containing the eval metrics in `.csv` files. |
| `plots_model_forecast_visual_usecase.ipynb` | Visual use case (Cyclone Yaku) model forecasts. |
| `plots_csi_lpips_per_timestep_plots.ipynb` | CSI and LPIPS per timestep ($/t$) plots used in the Research. |
| `plots_misc_research.ipynb` | Misc plots used in the research. |