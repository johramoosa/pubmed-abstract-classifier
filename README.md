# Biomedical Abstract Section Classifier

The experiment notebook currently relies on previously saved checkpoints
for some sections and does not yet support a complete run from scratch.
The inference notebook can run independently using the downloadable
exported model.

This project classifies sentences in biomedical abstracts into five sections: BACKGROUND, OBJECTIVE, METHODS, RESULTS, and CONCLUSIONS, using the PubMed 20k RCT dataset.

I compared majority-class and TF-IDF baselines with PyTorch neural models to establish reference performance and investigate the effects of model architecture, class weighting, sentence position, and fine-tuning.

The final model combines CNN text features with each sentence's relative position within its abstract. It accepts a complete abstract as an ordered list of sentences and predicts one section label per sentence.

## Experiments and results

All experiments used the official training, validation, and test splits. Model development and selection used validation results.

### Initial validation experiments

| Model | Validation accuracy (%) | Validation macro-F1 |
|---|---:|---:|
| Majority baseline | 32.98 | 0.0992 |
| TF-IDF + logistic regression | 80.92 | 0.7438 |
| Mean embedding classifier | 80.71 | 0.7426 |
| CNN | 81.11 | 0.7464 |
| Class-weighted CNN | 79.42 | 0.7360 |
| CNN + position, trained from scratch | 85.58 | 0.8018 |
| CNN + position, fine-tuned from the original CNN | 84.28 | 0.7882 |
| Position-only logistic regression | 68.78 | 0.5561 |

The fine-tuned model reused the original CNN's trained weights and updated all layers after adding the position feature. Its training budget therefore included the original CNN training and additional fine-tuning.

### Validation performance across three random seeds

The text-only CNN and CNN + position were evaluated using seeds 42, 123, and 2026. Values below are means ± sample standard deviations.

| Model | Validation accuracy (%) | Validation macro-F1 |
|---|---:|---:|
| CNN | 80.83 ± 0.33 | 0.7430 ± 0.0044 |
| CNN + position | 85.47 ± 0.15 | 0.7995 ± 0.0050 |

Key findings:

- Adding relative position improved accuracy and macro-F1 in all three seeded comparisons. Accuracy gains ranged from 4.42 to 5.02 percentage points, suggesting a consistent benefit across the seeds evaluated.
- Class weighting improved OBJECTIVE recall from 0.471 to 0.601 compared with the original CNN, but reduced overall validation accuracy and macro-F1.
- Fine-tuning the original CNN with the added position feature improved validation accuracy from 81.11% to 84.28%. However, the position-aware CNN trained from scratch performed better overall in these runs. The fine-tuned model achieved higher OBJECTIVE recall and F1 than the model trained from scratch.

### Final test evaluation

The final configuration was selected using validation results. Within that configuration, the checkpoint with the lowest validation loss across the three seeds was selected: CNN + position, seed 42, epoch 4.

| Model | Test accuracy (%) | Test macro-F1 |
|---|---:|---:|
| CNN + position | 85.24 | 0.8006 |

The test set was used for final evaluation, not model selection. Detailed classification reports, confusion matrices, and error analysis are provided in the experiment notebook.

## Limitations

- **BACKGROUND–OBJECTIVE confusion:** These classes remain difficult to distinguish. Some OBJECTIVE-labelled sentences read like background information, and both classes can occur near the beginning of an abstract.
- **Position requires abstract context:** The final model expects a complete abstract as an ordered list of sentences. Missing or reordered sentences change the relative-position feature.
- **Limited raw-text preprocessing:** The raw-text adapter adjusts some punctuation spacing but does not fully reproduce the dataset's preprocessing. It does not split paragraphs into sentences. Words absent from the training vocabulary map to an unknown token.
- **Generalization beyond the dataset:** Evaluation used PubMed 20k RCT. Performance on other document types, such as thesis abstracts or non-clinical research, has not been established.
- **Limited repeated-run evaluation:** The CNN and CNN + position were compared across three random seeds. Consistent gains across these runs support the observed benefit of position, but do not by themselves establish statistical significance.


## Installation and usage

### 1. Set up the project

The notebooks currently use Google Colab and Google Drive. Place the project folder at:

```text
/content/drive/MyDrive/Portfolio_Project/Abstract_Classifier
```

Alternatively, update `project_dir` and any remaining absolute paths in the notebooks to match your location.

The project folder should contain the `abstract_classifier/` Python package, both notebooks, and `requirements.txt`.

Mount Google Drive in Colab:

```python
from google.colab import drive

drive.mount("/content/drive")
```

Set the project path and make the package importable:

```python
from pathlib import Path
import sys

project_dir = Path(
    "/content/drive/MyDrive/Portfolio_Project/Abstract_Classifier"
)

if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))
```

Install dependencies if needed:

```python
%pip install -r /content/drive/MyDrive/Portfolio_Project/Abstract_Classifier/requirements.txt
```

The dependency versions were recorded from the working Colab environment. Restart the runtime if installation replaces packages that are already imported, then rerun the setup cells.

### 2. Run inference with the exported model

Place the exported model at:

```text
Abstract_Classifier/artifacts/abstract_classifier_v1.pt
```

Model weights are excluded from Git. A public download link has not yet been added. Until then, inference requires an existing copy of this artifact or completion of the training and export workflow.

Open `02_abstract_classifier_inference.ipynb` and run its cells in order. The notebook loads the model architecture, trained weights, vocabulary, and label mapping.

Provide a complete abstract as an ordered list of non-empty sentence strings:

```python
sentences = [
    "Effective treatments for persistent fatigue remain limited.",
    "This study aimed to evaluate whether an exercise programme reduces fatigue.",
    "We randomly assigned 350 patients to an exercise programme or usual care.",
    "Fatigue scores were measured at baseline and after 12 weeks.",
    "Patients in the exercise group had lower fatigue scores at 12 weeks.",
    "The exercise programme may help reduce persistent fatigue.",
]
```

After running the notebook's model-loading cell:

```python
from abstract_classifier.inference import predict_raw_abstract

predictions = predict_raw_abstract(
    sentences,
    model=model,
    vocab=vocab,
    id_to_label=id_to_label,
)

display(predictions)
```

The output includes the original sentence, prepared sentence, relative position, and predicted section label.

Use `predict_raw_abstract` for ordinary text. Use `predict_abstract` for sentences already formatted like the dataset. Both functions require sentences to be separated beforehand; neither automatically splits a paragraph.

The example above is illustrative and does not have independently verified ground-truth labels.

### 3. Explore or reproduce the experiments

Open `01_pubmed_rct_data_exploration.ipynb`.

The project uses the `PubMed_20k_RCT` directory from the PubMed RCT dataset repository, with the original numbers preserved. It does not use the variant that replaces numbers with an `@` symbol.

If the dataset is not already present, this Colab cell downloads it:

```python
from pathlib import Path
import subprocess

dataset_repo = Path("/content/pubmed-rct")

if not dataset_repo.exists():
    subprocess.run(
        [
            "git",
            "clone",
            "https://github.com/Franck-Dernoncourt/pubmed-rct.git",
            str(dataset_repo),
        ],
        check=True,
    )

data_dir = dataset_repo / "PubMed_20k_RCT"
```

Use the official files:

- `train.txt`: model fitting and training-data exploration.
- `dev.txt`: validation and model selection.
- `test.txt`: final evaluation.

The experiment notebook covers data checks, baselines, neural models, error analysis, repeated-seed experiments, final evaluation, and model export.

The resumable seeded experiments load existing checkpoints and train only unfinished epochs. Completed seeded runs skip further training. Earlier exploratory training cells may train again when executed.

Training checkpoints are stored under `checkpoints/`, and evaluation outputs are stored under `results/`. The final inference artifact is exported to `artifacts/abstract_classifier_v1.pt`.

The inference notebook does not require the dataset or training checkpoints once the exported artifact is available.


### Download the trained model

Download [abstract_classifier_v1.pt](https://drive.google.com/file/d/1hBNWh5w5653yAKwOgaFF09Xs3BluqgVd/view?usp=sharing) and place it in
the project's `artifacts/` folder.

For the documented Colab setup, the full path is:

`/content/drive/MyDrive/Portfolio_Project/Abstract_Classifier/artifacts/abstract_classifier_v1.pt`

Then run `02_abstract_classifier_inference.ipynb`.
No retraining is required.
