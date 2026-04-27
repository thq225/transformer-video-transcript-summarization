# Transformer-video-transcript-summarization

This project explores automatic summarization of meeting transcripts using Transformer-based models.
The system focuses on generating concise summaries from long meeting transcripts using a fine-tuned BART model.

The long-term goal is to build a video-to-summary pipeline, where a video recording is converted into a transcript and then summarized automatically.

# Project Overview
Meetings and presentations often produce long transcripts, making it difficult to quickly extract the key information.

This project addresses this problem by training a Transformer-based encoder–decoder model to generate summaries from meeting transcripts.

The model learns the conditional probability:

$P(summary \mid transcript, query)$

meaning the model generates a summary conditioned on the meeting transcript and the user query.

# Pipeline
The full system pipeline is designed as:
```
Video / Meeting Recording
        ↓
Speech-to-Text
        ↓
Meeting Transcript
        ↓
Transformer-based Summarization Model (BART)
        ↓
Concise Meeting Summary
```

## Data Source
This project uses the QMSum dataset (Query-based Meeting Summarization).

QMSum is a human-annotated benchmark dataset designed for meeting summarization tasks. It contains real-world meeting transcripts along with queries and corresponding reference summaries.

Each data sample includes:

* Transcript: a full meeting transcript
* Query: a question or focus point
* Reference Summary: a human-written summary relevant to the query

The dataset contains:

* 232 meetings
* 1,808 query–summary pairs
* Multiple domains (academic, business, etc.)

Dataset Access

You can download the dataset from:

https://github.com/Yale-LILY/QMSum

## Required Packages
See `environment.yml`

## How to Run

### 1. Prepare the data and model folders

Download the `qmsum` dataset folder and `outputs` folder from Google Drive:

https://drive.google.com/drive/folders/1_Xac-cEpGRe0xZBhJgf00TPFQsZqgMPM?usp=drive_link

### 2. Train the model
Open and run: 
```
main.ipynb
```

In the first setup cell, update the paths if needed:
```
DATA_DIR = Path("/content/drive/MyDrive/498data/qmsum")

OUTPUT_DIR = Path("/content/drive/MyDrive/498data/outputs/qmsum_large_bart")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
```

Then run all cells to train the model.

### 3. Run the full pipeline
To run the full pipeline from video to transcript to summary:
```
python full_pipeline.py
```

### 4. Run the web app
Install dependencies
```
pip install gradio transformers torch faster-whisper
```

Run the Gradio web app locally:
```
python app.py
```
Then open the local URL shown in the terminal, usually:
```
http://127.0.0.1:7860
```

### 5. Other notebooks
Other notebooks are used for testing, evaluation, and result visualization.
