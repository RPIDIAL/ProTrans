# ProTrans: Progression-aware Longitudinal Pretraining for Chest X-ray Analysis

Official PyTorch implementation of **ProTrans**, a progression-aware longitudinal pretraining framework for chest X-ray analysis.

---

## Data Preparation

### MIMIC-CXR-JPG

We use the MIMIC-CXR-JPG dataset as the image source. Please obtain access to MIMIC-CXR-JPG through PhysioNet: https://physionet.org/content/mimic-cxr-jpg/

### Progression Annotation

Disease progression information is obtained from Chest ImaGenome (https://physionet.org/content/chest-imagenome/1.0.0/) and further processed to construct longitudinal training samples.

### Generate Pretraining Samples

After preparing the datasets, run:

```bash
python preprocess_code/build_pretrain_samples.py
```

The generated samples will be saved as JSON files for pretraining. Each pretraining sample contains:

* Prior chest X-ray and radiology report
* Current chest X-ray and radiology report
* Progression discription


---

## Pretraining

Launch pretraining using:

```bash
bash scripts/1-1.mimic_train.sh
```

---

## Model Zoo

| Model    | Dataset   | Download    |
| -------- | --------- | ----------- |
| ProTrans | MIMIC-CXR | Coming Soon |


---

## Acknowledgements

This work is based on Diff-RRG, Med-ST, STG. We thank the creators and maintainers of these resources.

---
