# ERP-BatchFormer

`ERP-BatchFormer` is a modular, plug-and-play, template-guided multitask learning framework for deep-learning-based ERP classification. It is paired with `BGBank`, a complementary background-EEG-based data augmentation strategy.

The framework jointly optimizes a template regressor and an EEG classifier, with three key mechanisms:

- **BatchFormer-based template regression** — Transformer self-attention across the batch dimension models relationships among EEG trials within a mini-batch to estimate a representative ERP template for each trial.
- **SDTW-based temporal alignment** — soft Dynamic Time Warping supervises template regression, allowing temporal misalignment between predicted and reference templates while preserving multichannel ERP structure.
- **Learnable weighted fusion** — predicted templates are fused with the original EEG signals via learnable weights, and fed to the classifier for discriminative decoding.

## BGBank

A background EEG bank built by subtracting each trial's class-specific ERP template. The extracted background components and their transformed variants (TimeReverse, FTSurrogate) are stored and recombined with ERP templates to synthesize augmented samples. Applied only during subject-specific fine-tuning.

## Datasets

| Dataset   | Paradigm                                                                 | Subjects |
| --------- | ------------------------------------------------------------------------ | -------- |
| Dataset 1 | SSVEP ([Benchmark SSVEP](https://bci.med.tsinghua.edu.cn/download.html)) | 35       |
| Dataset 2 | P300 ([Brain Invaders](https://zenodo.org/records/2649069))              | 24       |

Backbones evaluated: **EEGNet**, **ShallowConvNet**, **DeepConvNet**.

## Key Results

- ERP-BatchFormer significantly improves accuracy over standalone backbones across **both datasets, all three backbones, and all tested signal lengths**, with absolute gains of **1.63–14.36 percentage points** (paired *t*-test, mostly *P* < 0.001).
- BGBank provides further complementary gains on top of ERP-BatchFormer.
- Consistent gains at short signal lengths highlight the framework's suitability for short-window ERP decoding.

## Requirements

- Python 3.9.19
- PyTorch 2.5.0
- CUDA 12.4
- GPU used in the paper: NVIDIA GeForce RTX 4060 Ti
