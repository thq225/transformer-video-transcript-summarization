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