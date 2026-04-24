# SMS Tone Reproduction Project

> Reproducing and analyzing tones in authentic and generated text messages.

**Author:** Camila Calvino · Montclair State University  
**Course:** APLN 552 
**Last Updated:** April 2026

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Reproducing Key Results](#reproducing-key-results)
- [Evaluation](#evaluation)

---

## Project Overview

This project investigates the reproduction of **SMS tone**

I explore whether language models can reliably reproduce a user's SMS tone given an authentic incoming utterance as input. Outputs are evaluated by human annotators. Post-annotation, perceived tone can be averaged to compute message-level and user-level tone scores.

**Key research questions and goals:**
- Can a LLM successfully reproduce the texting tone of a bilingual communicator through prompting or fine-tuning?
- Goal 1: Generate text message responses to real-incoming messages for multiple users
- Goal 2: Analyze the tones used in user’s responses and compare them to the tones of the generated responses

---

## Repository Structure

```
Text_Tone_Reproduction_Project/
├── README.md                  # This file
├── DATA.md                    # Data sources, licenses, privacy notes
│
├── data/
│   └── raw/                   # Original, unmodified datasets (see DATA.md)
│
├── notebooks/
│   ├── preprocessing_script.ipynb              # Splits the raw_data file into train/test splits for each user
│   ├── llama_3_few_shot_prompt.ipynb           # Constructs prompt with examples from training data, initiates generation
│   ├── llama_3_few_lora_fine_tuning.ipynb      # Constructs prompt and user adapters from training data, initiates generation
│   └── stylometric_analysis.ipynb              # Conducts stylometric analysis between authentic and generated data
│
└── results/
    └── outputs/               # Generated SMS-tone samples
```

---

### API Keys

For access to Llama 3 8B through HuggingFace API, you will first need to request access here (usually approved with 48 hours): https://huggingface.co/meta-llama/Meta-Llama-3-8B

### BYT Corpus

This project used a subset of data sourced from the Spanish-English Bilingual Youth Texts corpus (McSweeney 2016). 
Access to the BYTs dataset can be gained by going to this site: https://byts.commons.gc.cuny.edu/ The dataset is also publically available as a .csv and .sql file here: https://academicworks.cuny.edu/gc_etds/1459/

---

## Reproducing Key Outputs

Follow these steps in order to reproduce the project's main outputs. 

All scripts are set up to read files from a Google Drive. To begin, import raw_data.csv to the Google Drive directory you will work out of.


### Step 1 — Preprocess the Data

Import and open the notebook in the directory your data is in. Mount your drive and update your file paths as needed.
Run the notebooks cells.

**Expected output:** `data/train.csv` and `data/train.csv`

==================================================
  Split Summary
==================================================
  Total train rows : 2109
  Total test rows  : 523
  Users in train   : 7
  Users in test    : 7

          train  test  total  test_%
user_id                            
U01        148    36    184    19.6
U03         30     7     37    18.9
U09        176    43    219    19.6
U11         92    23    115    20.0
U12        112    28    140    20.0
U14         55    13     68    19.1
U15       1496   373   1869    20.0
==================================================

---

### Step 2 — Prompt / Fine-tune & Generate

Prompt building and/or fine-tuning and generation occur in the same respective scripts. The few shot prompt and fine tuning notebooks can be run in any order. The prompts can be adjusted within the prompt builder function - it is recommended the prompts used for both approaches are kept constant if you are doing a comparitice analysis.

Import and open the notebook in the directory your data is in. Mount your drive and update your file paths as needed.
Run the notebooks cells.

> **Note:** The notebooks were run using a T4 GPU runtime. Even with this runtime, it is recommended to generate messages in batches (ie. 3 users at a time) especially for fine-tuning as Google Colab can time out causing you to lose progress. Be sure to append to the combined csv file (or rename the output file) if batching this way so as to not overwrite the file from a previous batch.

---

### Step 3 — Annotation and Analysis

The tone analysis portion cannot be reproduced without a tone evaluation subprocess. For my project, this was done by human annotators. A template of an annotation sheet is provided under the annotation folder in this repository. If human annotation is not ppossible, this could potentially be mitigated through the use of an LLM as a judge.

The stylometric analysis does not require human annotation as it only evaluates form. The stylometric analysis notebook is available under the codes folder. Features can be edited and omitted as desired. This notebook will produce a bar chart, heat maps, and violin plots. Additionally, a report of statistically significant feature differences is outputted.

---


## License

[Insert license, e.g., MIT / Apache 2.0 / CC BY 4.0]