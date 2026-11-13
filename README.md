# UT-EndoMRI-project

Cross-Site Generalization of Pelvic MRI Segmentation (UT-EndoMRI)
Code for a thesis study on cross-site generalization and data pooling for deep-learning segmentation of the uterus, ovaries and endometriomas on T2-weighted pelvic MRI, plus an analysis of inter-rater variability and its effect on model training and evaluation.
A small 2D U-Net is trained on one site, the other site, or both pooled, and every model is tested on the same held-out patients from both sites.
Both notebooks are written for Google Colab with the data on Google Drive. They do not run as plain Python scripts without editing the paths and removing the drive.mount calls.
Contents
Notebook
	Purpose
	thesis_pooled_vs_crosssite_2-5.ipynb
	Main pipeline: A-only / B-only / pooled experiments, inter-rater analysis, rater-specific models, failure analysis, label audits, final tables and figures
	results_rater_visual_check_standalone.ipynb
	Standalone, inference-only notebook: visual comparison of rater annotations and model predictions, and rater-agreement statistics. 
	

Dataset and technology use
The code expects the UT-EndoMRI dataset in this layout:
UT-EndoMRI/
├── D1_MHS/                      # Site A: training / in-distribution site
│   ├── D1-000/
│   │   ├── D1-000_T2.nii.gz
│   │   ├── D1-000_ut_r1.nii.gz  # uterus, rater 1
│   │   ├── D1-000_ut_r2.nii.gz  # uterus, rater 2 (if present)
│   │   ├── D1-000_ov_r1.nii.gz  # ovary
│   │   └── D1-000_em_r1.nii.gz  # endometrioma
│   └── ...
└── D2_TCPW/                     # Site B: external site
    ├── D2-006/
    │   ├── D2-006_T2.nii.gz
    │   ├── D2-006_ut.nii.gz     # single rater, no _r suffix
    │   └── ...
    └── ...
* Site A (D1_MHS) has multiple raters (r1, r2, r3, ...). Site B (D2_TCPW) has a single rater.
* Classes: 0 background, 1 uterus (ut), 2 ovary (ov), 3 endometrioma (em).
* T2 only is used at both sites (MRI_SEQUENCE_PRIORITY = ["T2"]) so image contrast is matched. There is deliberately no fallback sequence; patients without T2 are excluded.
The dataset itself is not included in this repository. Please obtain it from its original source and cite it accordingly. The dataset can be found here: https://zenodo.org/records/13749613 
For debugging, code formatting, creating this repository, and help coding, Claude (Anthropic, 2026) was used. After using this tool, the author reviewed and edited the content as needed and takes full responsibility for the content of the code and repository. 


Pipeline overview
1. Configuration
A central config cell defines paths (DATA_ROOT, SITE_A_DIR, SITE_B_DIR), sequence selection, structure-to-class mapping, and hyperparameters.
Setting
	Value
	Image size
	256 × 256
	Slice axis
	2 (axial, after canonical reorientation)
	Min. foreground pixels per slice
	10
	Batch size
	8
	Epochs
	40
	Learning rate
	1e-3 (Adam, ReduceLROnPlateau, factor 0.5, patience 4)
	U-Net base channels
	32
	Split (per site, case level)
	65% train / 15% val / 20% test
	Seed
	42 (list in SEEDS)
	2. Data loading and preprocessing
* discover_cases walks a site folder and builds one record per patient (image path + per-structure label paths). Both label naming conventions (multi-rater _r1 and single-rater) are handled; PREFERRED_RATER = "r1" is used when several exist.
* load_combined_mask merges per-structure labels into one multi-class mask. Labels saved on a different voxel grid are resampled onto the image grid with nearest-neighbour interpolation using the NIfTI affines. Later structures are painted on top of earlier ones where they overlap.
* Volumes are reoriented to canonical RAS+ (nib.as_closest_canonical) before slicing, so the slice axis is anatomically consistent across patients.
* Slices with fewer than MIN_FOREGROUND_PIXELS labelled pixels are skipped.
* Each slice is percentile-clipped (1st–99th), min–max normalised to [0, 1], and resized to 256 × 256 (bilinear for images, nearest for masks). Training augmentation is a random horizontal flip.
* EndoMRISliceDataset serves 2D slices as (image, mask, meta).
3. Model and loss
* SmallUNet: a from-scratch 2D U-Net with 4 encoder stages, a bottleneck, and 4 decoder stages with skip connections (Conv-BatchNorm-ReLU blocks, transposed-convolution upsampling).
* DiceCELoss: weighted sum of cross-entropy and soft Dice loss (ce_weight = 0.5).
* dice_per_class: per-class Dice that returns NaN for classes absent from both prediction and reference, so absent classes are not counted as perfect scores.
4. Main experiments (run_all)
Three models are trained, each tested on the same A_test and B_test patients:
Experiment
	Train
	Validation
	Notes
	A_only
	A_train
	A_val
	A in-site, A→B cross-site
	B_only
	B_train
	B_val
	B in-site, B→A cross-site
	pooled
	A_train + B_train
	A_val + B_val
	

	Safeguards against data leakage:
Splits are made at the case level
check_no_leaks verifies that train/val/test are disjoint by case ID, file path and volume hash; check_slice_level repeats the check at slice level.
train_model never receives test data. Checkpoints are selected on validation foreground Dice only.
Outputs: unet_<exp>_seed<seed>.pt, history_<exp>_seed<seed>.csv, per_case_metrics_all.csv, splits.csv.
5. Inter-rater agreement (site A)
* Pairwise 3D Dice between every rater pair, using all patients both raters annotated.
* Volume comparison per rater, with paired Wilcoxon tests, Holm correction, ICC(2,1), and a Friedman test across all raters.
* 2D protocol agreement: one rater plays "prediction" and the other "reference" (both directions averaged), using the same slice selection, resolution and Dice definition as the model results. 
Missing-label rules, applied consistently throughout:
* Uterus: every patient has one, so a missing (or empty) uterus file means not annotated. That rater is excluded for that patient, never scored as an empty mask.
* Ovaries / endometrioma: for patients a rater annotated, a missing file counts as an empty mask. If both raters are empty, the patient has no value for that organ.
6. Rater-specific models
One A-only model is trained per rater (RUN_RATER_MODELS), all on the same patients and split, so only the labels differ. Each is evaluated against every rater's labels, giving a trained-on × evaluated-against matrix. The diagonal is "model vs. the rater it learned from"; the off-diagonal is "model vs. an unseen rater". Evaluation is volumetric (3D Dice per patient) over the union of labelled slices (EVAL_SLICE_MODE = "union") or all slices ("all"). Raters with too few annotated patients (MIN_RATER_PATIENTS = 15) are not used for training but are still used as evaluation references.
7. Failure analysis and label audits
* Per-case covariates (voxel spacing, image shape, intensity statistics, contrast proxy, foreground fraction) are computed and related to Dice (Spearman correlation) to look for causes of poor generalization.
* An audit of label files finds files that exist but are empty, rebuilds the patient flow, and checks that saved results follow the uterus rule. A corrected results CSV is written; the original is untouched. 
* A cell reports how much training data had missing labels per site and split.
* Patients that scored near 0 uterus Dice under every model (CHECK_IDS) are inspected individually.
8. Final tables and figures (run_final_tables)
Output
	Content
	Table 1
	Per-organ Dice per model and test set
	Table 2
	RQ2 generalization gaps and RQ3 pooling comparisons (Holm-corrected over 12 tests)
	Table 3
	Data/label availability, training exposure, eligible test patients
	Table 4
	Pairwise rater agreement (3D Dice)
	Appendix
	Foreground mean; uncorrected vs. corrected means
	Tables are saved as CSV, final cells draw the forest plots and the model-by-rater Dice heatmaps (model_dice_heatmaps.png/.pdf). Those plotting cells use hard-coded values (mean, CI, n), not a live read of the results files, so update them if you re-run the experiments.
Statistics used
* The unit of analysis is the patient (seeds averaged per patient first).
* Confidence interval of a mean: Student-t by default (CI_METHOD = "t")
* Paired differences (same patients, two models): exact Wilcoxon signed-rank test, requiring at least 5 pairs.
* RQ2 gaps compare different patients (in-site vs. other site), so a Welch interval is reported with no p-value.
* Means below FLOOR = 0.10 are marked with *; differences near the floor are not interpretable.
9. Evaluation protocol
Unless stated otherwise: keep slices whose reference mask has at least MIN_FOREGROUND_PIXELS labelled pixels, resize masks to 256 × 256 (nearest neighbour), compute Dice per class per slice (classes absent from both masks are skipped), average over slices within a patient, then over patients.
Standalone visual-check notebook
results_rater_visual_check_standalone.ipynb contains copies of the config, data-loader, model and loss/metric cells, so it can be run without the main notebook. No training is run. It needs a trained checkpoint on Drive.
It produces:
* Rater vs. rater vs. model figures. Each row shows image | r1 | r2 | r3 | agreement map | model prediction. In the agreement map, yellow marks pixels where annotating raters disagree (including "one rater left it out"). Dice vs. each rater is printed under the panels, and each patient is labelled with its split (train/val/test), since showing training patients flatters the model.
* Complete-patient comparison (patients annotated by all raters): overlays zoomed on the labelled region, per-organ pairwise 3D Dice and per-rater volume plots, and a list of patients where raters disagree most.
* Pairwise rater agreement (3D and 2D protocol) as described above.
* A label-file check that classifies each rater's uterus file as ok, file is EMPTY, LOST in resampling, or NO FILE.
* Rater-specific model training (resumable after a Colab disconnect).
Edit the settings block at the top of each cell (FOLDER_NAME, CKPT_NAME, CASE_IDS, SPLIT_FILTER, N_PATIENTS, ...).
Requirements
Results were produced with the following versions (Google Colab runtime). A GPU is recommended; a CPU works for small tests.
Package
	Version
	Used for
	Python
	3.13.16
	

	PyTorch (torch)
	2.14.1+cu130
	model, training
	NumPy
	2.4.6
	arrays
	pandas
	2.2.3
	tables and CSVs
	nibabel
	5.4.2
	reading NIfTI files, resampling
	scikit-image (skimage)
	0.25.2
	resizing slices
	SciPy
	1.16.3
	statistical tests
	Matplotlib
	3.10.0
	figures
	seaborn
	0.13.2
	heatmaps and plots
	openpyxl
	not recorded
	optional Excel export of tables
	google.colab is provided by the Colab runtime and is not installed separately. The remaining imports (os, glob, re, json, hashlib, itertools, random, collections) are part of the Python standard library.
Install with:
bash
pip install -r requirements.txt
The +cu130 suffix on PyTorch means a CUDA 13.0 build. For a GPU install, follow the selector on pytorch.org; requirements.txt pins the base version only.
Running on Colab
1. Upload the UT-EndoMRI dataset to Google Drive.
2. Set DATA_ROOT in the config cell.
3. Run cells top to bottom. Config → data loader → model/loss → experiment cells must run first, because later analysis cells reuse their functions and variables (discover_cases, load_combined_mask, SmallUNet, constants, ...).
4. Analysis cells locate their input files by name on Drive (find_folder, newest, glob patterns), so keep the output files from earlier steps in your project folder.
Typical run order (main notebook)
1. Mount Drive, run config, data-loader, model and loss cells.
2. Run the experiment-design cell, then run_all() for A-only / B-only / pooled.
3. Inter-rater agreement (run_inter_rater), then rater-specific models (run_rater_experiments).
4. Failure analysis, label audits, availability tables.
5. run_final_tables() and the model-vs-rater evaluation, then the plotting cells.
Reproducibility
* Seeds are set for random, NumPy and PyTorch (SEEDS = [42]). Add more seeds (e.g. [42, 43, 44, 45, 46]) to average over several random splits.
* The split function is shared by all experiments, so the same seed gives the same patient split everywhere (splits.csv / splits-2.csv record it).
* GPU training is not bit-for-bit deterministic, so small numerical differences between runs are expected.
Known limitations
* A missing label file is treated as "structure absent" for ovaries and endometrioma, but absence of a file is not confirmed absence of the structure.
* Models were trained with some structures as background where label files were missing or empty (see the label audit). This is reported as a limitation, not corrected.
* Test sets are small, which is why exact tests and t-intervals are used and why differences near the Dice floor are flagged.
* The model is 2D and slice-based, with no 3D context.
License and citation


Liang, X., Alpuing Radilla, L. A., Khalaj, K., Mokashi, C., Guan, X., Roberts, K. E., Sheth, S. A., Tammisetti, V. S., & Giancardo, L. (2024). UTHealth - Endometriosis MRI Dataset (UT-EndoMRI) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.13749613


Anthropic. (2026). Claude Sonnet 5.5 [Large language model]. https://claude.ai
