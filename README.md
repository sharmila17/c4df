# C4DF: Confidence-Calibrated Contextual Control for Diverse and Faithful Image Captioning

This repository contains the research implementation associated with
**C4DF (Confidence-Calibrated Contextual Control for Diverse and
Faithful Image Captioning)**.

C4DF is a decoding-time framework designed to reduce visually
unsupported content while preserving diversity in generated image
captions. The implementation uses a ResNet-50 visual encoder, a
Transformer-based caption decoder, prefix-based diversity control, a
CLIP-derived Visual Semantic Prior, and confidence-calibrated token
suppression.

## Repository Contents

-   `c4df1.ipynb` --- model preparation/training, vocabulary
    construction, checkpointing, and diversity-token training workflow.
-   `c4df2.ipynb` --- C4DF decoding, Visual Semantic Prior
    construction, confidence calibration, evaluation metrics, ablation
    experiments, and qualitative evaluation.
-   `labels.txt` --- 1,500-word label bank used by the Visual Semantic
    Prior.
-   `requirements.txt` --- Python package requirements.

The notebooks are preserved as the original research notebooks used
during development. They contain intermediate experimental/debugging
cells in addition to the final implementation and are **not intended to
be executed blindly using "Run All"**. Follow the workflow below and use
the final experiment cells/configuration.

## Method Configuration

The final C4DF experiments use the following principal settings:

-   Visual backbone: ImageNet-pretrained ResNet-50
-   Image input size: 224 × 224
-   Transformer embedding dimension: 512
-   Diversity tokens: `<DIV0>`, `<DIV1>`, `<DIV2>`
-   Visual Semantic Prior: frozen CLIP image/text embeddings
-   Label-bank size: 1,500 words
-   Context-aligned labels (Top-N): 40
-   Maximum suppression strength (`alpha`): 0.8
-   Confidence-calibration threshold (`tau`): 0
-   Sigmoid confidence slope (`k`): 25

The notebooks also contain ablation experiments for suppression
strength, confidence slope, and fixed versus confidence-calibrated
suppression.

## Dataset

The experiments use the **MS-COCO** dataset. The dataset is not
redistributed in this repository.

Download the required images and caption annotations from the official
MS-COCO distribution and update the dataset paths in the notebooks for
your local installation. The training notebook uses COCO training
images/captions, while evaluation uses the corresponding validation data
specified by the experiment cells.

A typical local layout is:

``` text
COCO/
├── train2014/
├── val2014/ or val2017/
└── annotations/
    ├── captions_train2014.json
    └── captions_val*.json
```

Use the exact split referenced by the relevant notebook cell when
reproducing a particular experiment.

## Additional Required Artifacts

To reproduce inference from the trained model, the following artifacts
are required:

-   `vocab.pt` --- vocabulary generated during model
    preparation/training.
-   `checkpoint_best.pt` --- trained checkpoint used for the final C4DF
    evaluation.
-   `labels.txt` --- included in this repository.

The trained checkpoint is not stored directly in the Git repository
because of its size. Add the public checkpoint location here when it is
uploaded:

``` text
Checkpoint: <[PUBLIC CHECKPOINT LINK](https://github.com/sharmila17/c4df/releases/download/c4df/checkpoint_best.pt)>
```

Similarly, if `vocab.pt` is distributed separately:

``` text
Vocabulary: <PUBLIC VOCAB LINK OR REPOSITORY PATH>
```

## Environment Setup

A Python environment with PyTorch and the packages listed in
`requirements.txt` is required.

``` bash
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended for training and large-scale
generation. CPU execution may be considerably slower.

## Reproduction Workflow

### 1. Prepare MS-COCO

Download MS-COCO images and annotations and modify the dataset paths in
the notebooks to match your local directory structure.

### 2. Prepare the label bank

Keep `labels.txt` accessible to the evaluation notebook and update its
path if necessary. This file is used to construct the CLIP-based
image-specific semantic context.

### 3. Training / model preparation

Open:

``` text
c4df1.ipynb
```

This notebook contains vocabulary construction, model definition,
training, checkpointing, and diversity-prefix conditioning. Intermediate
development cells are retained for provenance; use the final
training/resume workflow rather than executing every historical cell
sequentially.

If reproducing inference using the released checkpoint, full retraining
is not required.

### 4. C4DF decoding and evaluation

Open:

``` text
c4df2.ipynb
```

Configure the paths to:

-   MS-COCO images and annotations
-   `checkpoint_best.pt`
-   `vocab.pt`
-   `labels.txt`

Use the final C4DF configuration:

``` text
Top-N = 40
alpha = 0.8
tau   = 0
k     = 25
```

The notebook contains the C4DF decoding pipeline and evaluation
procedures.

## Evaluation

The implementation includes evaluation for standard captioning quality,
faithfulness, and diversity. Depending on the experiment cells, these
include:

-   BLEU
-   ROUGE-L
-   CIDEr
-   CHAIRi
-   CHAIRs
-   Object Precision
-   Noun Non-Context Rate (NounNCR)
-   Distinct-1 / Distinct-2

NounNCR evaluates whether noun-like generated content is consistent with
the image-specific semantic context obtained from the Visual Semantic
Prior. It is intended as a complementary measure to established
object-hallucination metrics such as CHAIR.

## Important Reproducibility Note

The notebooks reflect the original experimental workflow and therefore
retain exploratory, intermediate, and superseded cells. Some
early/debugging cells may depend on variables created interactively or
contain placeholders that were superseded by later implementations. For
reproducibility:

1.  Do not use notebook-wide `Run All` without reviewing the cell
    sequence.
2.  Use the final training and evaluation sections/configurations
    described above.
3.  Update all local Windows paths before execution.
4.  Use the released `checkpoint_best.pt`, `vocab.pt`, and the provided
    `labels.txt` when reproducing the reported decoding experiments.
5.  Keep random seeds and decoding parameters unchanged when attempting
    direct comparisons.

## Citation

If you use this implementation in academic work, please cite the
associated C4DF article. The final bibliographic citation can be added
here after publication:

``` text
<C4DF ARTICLE CITATION>
```

## Resource Availability

MS-COCO is an external public dataset and is not redistributed with this
repository. The implementation notebooks and C4DF label bank are
provided here to support reproducibility. The trained model checkpoint
should be accessed using the checkpoint link provided above once
released.

## License

Add the selected repository license in a `LICENSE` file before public
release. Dataset and third-party model components remain subject to
their respective licenses.
