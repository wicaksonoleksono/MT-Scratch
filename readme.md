

# BI-GRU Encoder Decoder

This repository implements a machine translation system using a Bidirectional GRU (BI-GRU) encoder and a standard GRU decoder.

## Overview

The system leverages the strengths of bidirectional recurrent networks to capture context from both past and future tokens in the source language while generating translations step-by-step with a unidirectional decoder. This combination allows the model to effectively understand the source sentence and produce fluent target sentences.

## Model Architecture

### BI-GRU Encoder
- **Bidirectional Processing:** The encoder processes the input sequence in both forward and backward directions using a GRU. This ensures that the hidden representations contain information from both the beginning and the end of the sentence.
- **Contextual Representation:** The outputs from both directions are typically concatenated (or merged) to form a comprehensive context vector that captures rich information about the entire sentence.
- **Encoder Outputs:** These outputs, along with the final hidden states, serve as the foundation for the decoder’s translation process.

### GRU Decoder
- **Sequential Generation:** The decoder is a unidirectional GRU that generates the target language sentence token-by-token.
- **Token-by-Token Prediction:** At each time step, the decoder uses the previous token (starting with the start-of-sentence token) and its current hidden state to predict the next token.
- **Bridge Mechanism (if applicable):** In some cases, a transformation (often called a "bridge") is applied to the encoder’s final hidden state to initialize the decoder’s hidden state, ensuring compatibility between the two modules.
- **Stop Criterion:** The decoding process continues until an end-of-sentence token is generated or a predefined maximum length is reached.

## Notebooks for Workflow

To work with this project, follow the sequence outlined in the following notebooks:

### Data Preprocessing
- **Notebook:** `preproc.ipynb`
- **Purpose:** This notebook handles all data cleaning, tokenization, and the creation of input/output dictionaries. It ensures that your dataset is correctly formatted and ready for training.
- **Steps Covered:**
  - Cleaning raw text data.
  - Tokenizing sentences.
  - Creating vocabulary mappings for both source and target languages.

### Training Experiments
- **Notebook:** `05.experiments_BiGRU.ipynb`
- **Purpose:** This notebook demonstrates how to set up and run training experiments using the BI-GRU encoder and GRU decoder.
- **Steps Covered:**
  - Initializing the encoder and decoder.
  - Configuring the training loop, including loss computation and backpropagation.
  - Monitoring training performance and saving model checkpoints.
- **Note:** Adjust hyperparameters as necessary to optimize performance on your specific dataset.

### Translation and Testing
- **Notebook:** `06.translateandtest_BiGRU.ipynb`
- **Purpose:** Once the model is trained, use this notebook to translate new sentences and evaluate translation quality.
- **Steps Covered:**
  - Loading the pre-trained model.
  - Translating input sentences with the BI-GRU encoder and GRU decoder.
  - Comparing the predicted translations against ground truth.
  - Evaluating using relevant translation metrics.

## How to Use

1. **Preprocess Your Data:**
   - Start by running `preproc.ipynb` to prepare your dataset.
   
2. **Train the Model:**
   - Execute `05.experiments_BiGRU.ipynb` to train the BI-GRU encoder and GRU decoder on your dataset.
   - Monitor the training process and adjust parameters as needed.
   
3. **Translate and Test:**
   - Use `06.translateandtest_BiGRU.ipynb` to translate new sentences and evaluate the model's performance.
   - Analyze the output and iterate on the training process if necessary.

## Requirements
- export it from yml (conda)
- conda env create -f environment.yml

- Python 3.x
- [PyTorch](https://pytorch.org/)
- Other dependencies: numpy, pandas, etc. (see `requirements.txt` for a complete list)

