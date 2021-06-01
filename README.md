# pos-tagging-elmo
A simple POS-tagging classifier using ELMo or fastText embeddings

The code supports training a classifier using your own ELMo embeddings, ELMo embeddings from ElmoForManyLangs project (EFML) or fastText embeddings.
Any other static embeddings (Glove, word2vec, etc.) should work if provided in the same text (.vec) format as fastText.

Cross-lingual usage is supported by means of training the classifier on any one language, then applying the vector mappings from second (evaluation) language to the train language  at evaluation time.
Three mapping approaches are supported: `vecmap` (https://github.com/artetxem/vecmap), `MUSE` (https://github.com/facebookresearch/MUSE), and `ELMoGAN` (https://github.com/MatejUlcar/elmogan). 
Please refer to https://github.com/MatejUlcar/vecmap-changes for extracting (saving) the final matrices used for vecmap mapping.

## Usage

**(Optional) step 0:**
For faster training, you can pre-embed the datasets, using `pre-embed_fasttext.py`, `pre-embed_efml.py` or `pre-embed_elmo.py` depending on the embeddings you use.

**Step 1:**
Train the classifier using `train_fasttext.arc3.py`, `train_efml.arc3.py` or `train_elmo.arc3.py`. Apply the `--help` flag to check all the parameters. If you pre-embedded the dataset (step 0), do not add the
embeddings (weights/options) files at this stage (ie. do not use those flags). If using vecmap mapping approach, provide the necessary mapping files at the training stage already, as vecmap generally
changes both source and target language vectors.

**Step 2:**
Evaluate the classifier using `predict_fasttext.py`, `predict_efml.py` or  `predict_elmo.py`. Provide the necessary mapping files if using any cross-lingual approach.

## Requirements
The following python3 packages are required to train and evaluate the POS-tagging classifier:

`numpy`
`keras`
`sklearn`
`tensorflow` (as keras backend)
`allennlp==0.9.0` (ELMo only)
`elmoformanylangs` (ElmoForManyLangs only)
