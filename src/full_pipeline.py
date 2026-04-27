import re
import time
from pathlib import Path

import torch
from faster_whisper import WhisperModel
from transformers import BartTokenizerFast, BartForConditionalGeneration

# =========================
# Config
# =========================
# VIDEO_PATH = "/content/drive/MyDrive/498data/videos/milestone1.mp4"

# MODEL_DIR = "/content/drive/MyDrive/498data/outputs/qmsum_bart_baseline/best_model"


# or local:

VIDEO_PATH = "data/videos/milestone1.mp4"
MODEL_DIR = "outputs/qmsum_bart_baseline/best_model"

TRANSCRIPT_OUT = "transcript.txt"
SUMMARY_OUT = "summary.txt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 1. Video -> Transcript
# =========================
print("Loading Whisper model...")

whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if whisper_device == "cuda" else "int8"

whisper_model = WhisperModel(
    "small",
    device=whisper_device,
    compute_type=compute_type
)

print("Transcribing video...")
segments, info = whisper_model.transcribe(VIDEO_PATH, language="en")

transcript = ""
for segment in segments:
    transcript += segment.text.strip() + " "

transcript = re.sub(r"\s+", " ", transcript).strip()

with open(TRANSCRIPT_OUT, "w", encoding="utf-8") as f:
    f.write(transcript)

print("Transcript saved.")
print("Transcript preview:")
print(transcript[:500])

# =========================
# 2. Load BART model
# =========================
print("\nLoading BART model...")

tokenizer = BartTokenizerFast.from_pretrained(MODEL_DIR)
model = BartForConditionalGeneration.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

print("BART model loaded on:", DEVICE)

# =========================
# 3. Helpers
# =========================
def clean_text(text):
    text = text.replace(">>", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_into_sentence_chunks(text, tokenizer, max_tokens=400):
    text = clean_text(text)

    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current = []

    for sent in sentences:
        candidate = " ".join(current + [sent])
        token_count = len(tokenizer.encode(candidate, add_special_tokens=False))

        if token_count > max_tokens and current:
            chunks.append(" ".join(current))
            current = [sent]
        else:
            current.append(sent)

    if current:
        chunks.append(" ".join(current))

    return chunks


def generate_bart_summary(query, text, tokenizer, model,
                          input_max_length=512,
                          output_max_length=120,
                          output_min_length=35):
    input_text = f"query: {query}\n\ntranscript:\n{text}"

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=input_max_length,
        truncation=True
    ).to(DEVICE)

    with torch.no_grad():
        summary_ids = model.generate(
            **inputs,
            max_length=output_max_length,
            min_length=output_min_length,
            num_beams=4,
            no_repeat_ngram_size=4,
            repetition_penalty=1.5,
            length_penalty=1.0,
            early_stopping=True
        )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()

    if not summary.endswith((".", "!", "?")):
        last_punc = max(summary.rfind("."), summary.rfind("!"), summary.rfind("?"))
        if last_punc > 40:
            summary = summary[:last_punc + 1]

    return summary


# =========================
# 4. Transcript -> Summary
# =========================
print("\nSplitting transcript into chunks...")
chunks = split_into_sentence_chunks(transcript, tokenizer, max_tokens=400)

print(f"Number of chunks: {len(chunks)}")

section_query = "Write a concise summary of the important points in this section."

chunk_summaries = []

for i, chunk in enumerate(chunks):
    print(f"Summarizing chunk {i+1}/{len(chunks)}...")

    chunk_summary = generate_bart_summary(
        query=section_query,
        text=chunk,
        tokenizer=tokenizer,
        model=model,
        input_max_length=512,
        output_max_length=110,
        output_min_length=30
    )

    chunk_summaries.append(chunk_summary)

# Combine section summaries
combined_text = "\n".join(
    [f"Section {i+1}: {summary}" for i, summary in enumerate(chunk_summaries)]
)

final_query = """
Summarize this project presentation clearly.
Focus on the problem, project goal, dataset, model, training setup, results, and future work.
"""

print("\nGenerating final summary...")

final_summary = generate_bart_summary(
    query=final_query,
    text=combined_text,
    tokenizer=tokenizer,
    model=model,
    input_max_length=1024,
    output_max_length=260,
    output_min_length=80
)

with open(SUMMARY_OUT, "w", encoding="utf-8") as f:
    f.write(final_summary)

print("\n===== FINAL SUMMARY =====\n")
print(final_summary)

print("\nSaved transcript to:", TRANSCRIPT_OUT)
print("Saved summary to:", SUMMARY_OUT)
