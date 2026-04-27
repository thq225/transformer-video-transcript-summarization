import re
import torch
import gradio as gr
from faster_whisper import WhisperModel
from transformers import BartTokenizerFast, BartForConditionalGeneration

# =========================
# Model repos
# =========================
SMALL_BART_DIR = "thoiquach/qmsum-bart-summarizer"
LARGE_BART_DIR = "thoiquach/qmsum-bart-large-summarizer"

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Load models
# =========================
print("Using device:", device)

print("Loading Whisper model...")
whisper_model = WhisperModel(
    "small",
    device="cuda" if device == "cuda" else "cpu",
    compute_type="float16" if device == "cuda" else "int8",
)

print("Loading small BART model...")
small_tokenizer = BartTokenizerFast.from_pretrained(SMALL_BART_DIR)
small_bart = BartForConditionalGeneration.from_pretrained(SMALL_BART_DIR).to(device)
small_bart.eval()

print("Loading large BART model...")
large_tokenizer = BartTokenizerFast.from_pretrained(LARGE_BART_DIR)
large_bart = BartForConditionalGeneration.from_pretrained(LARGE_BART_DIR).to(device)
large_bart.eval()

if device == "cuda":
    small_bart = small_bart.half()
    large_bart = large_bart.half()


# =========================
# Helpers
# =========================
def clean_text(text):
    text = text.replace(">>", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def transcribe_video(video_path):
    segments, _ = whisper_model.transcribe(video_path, language="en")

    transcript = ""
    for segment in segments:
        transcript += segment.text.strip() + " "

    return clean_text(transcript)


def get_output_length(transcript, ratio):
    words = len(transcript.split())

    if ratio == "10%":
        target_words = int(words * 0.10)
    elif ratio == "20%":
        target_words = int(words * 0.20)
    else:
        target_words = int(words * 0.30)

    target_tokens = int(target_words * 1.3)

    max_len = min(max(target_tokens, 80), 512)
    min_len = max(30, int(max_len * 0.4))

    return max_len, min_len


def generate_summary(
    tokenizer,
    model,
    query,
    text,
    input_max_length=1024,
    output_max_length=180,
    output_min_length=50,
):
    input_text = f"query: {query}\n\ntranscript:\n{text}"

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=input_max_length,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        summary_ids = model.generate(
            **inputs,
            max_length=output_max_length,
            min_length=output_min_length,
            num_beams=4,
            no_repeat_ngram_size=4,
            repetition_penalty=1.6,
            length_penalty=1.0,
            early_stopping=True,
        )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()


def split_into_chunks(text, tokenizer, max_tokens=400):
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


def summarize_small_bart_with_chunking(transcript):
    chunks = split_into_chunks(transcript, small_tokenizer, max_tokens=400)

    section_query = "Write a concise summary of the important points in this section."

    chunk_summaries = []

    for chunk in chunks:
        summary = generate_summary(
            tokenizer=small_tokenizer,
            model=small_bart,
            query=section_query,
            text=chunk,
            input_max_length=512,
            output_max_length=110,
            output_min_length=30,
        )
        chunk_summaries.append(summary)

    combined_text = "\n\n".join(
    [
        f"Section {i + 1}:\n{summary}"
        for i, summary in enumerate(chunk_summaries)
    ]
)

    return combined_text

def select_large_summary(ratio, large_10, large_20, large_30):
    if ratio == "10%":
        return large_10
    elif ratio == "20%":
        return large_20
    else:
        return large_30

# =========================
# Main app function
# =========================
def process_video(video_path):
    if video_path is None:
        return "Please upload a video.", "", "", "", "", ""

    transcript = transcribe_video(video_path)

    query = (
        "Summarize this project presentation. Focus on the problem, goal, "
        "dataset, model, results, and future work."
    )

    small_no_chunk_summary = generate_summary(
        tokenizer=small_tokenizer,
        model=small_bart,
        query=query,
        text=transcript,
        input_max_length=512,
        output_max_length=160,
        output_min_length=50,
    )

    small_chunk_text = summarize_small_bart_with_chunking(transcript)

    # Generate all 3 large BART summaries once
    large_10 = generate_summary(
        large_tokenizer, large_bart, query, transcript,
        input_max_length=1024,
        output_max_length=130,
        output_min_length=50,
    )

    large_20 = generate_summary(
        large_tokenizer, large_bart, query, transcript,
        input_max_length=1024,
        output_max_length=220,
        output_min_length=80,
    )

    large_30 = generate_summary(
        large_tokenizer, large_bart, query, transcript,
        input_max_length=1024,
        output_max_length=320,
        output_min_length=120,
    )

    # Default show 20%
    return (
        transcript,
        large_20,
        small_no_chunk_summary,
        small_chunk_text,
        large_10,
        large_20,
        large_30,
    )

# =========================
# Gradio UI
# =========================
with gr.Blocks() as demo:
    gr.Markdown("# Video Transcript Summarization Demo")

    video_input = gr.Video(label="Upload MP4 Video")
    run_btn = gr.Button("Generate Summary")

    gr.Markdown("## Transcript")
    transcript_box = gr.Textbox(lines=10, label="Transcript")

    gr.Markdown("## Large BART Summary (Final Result)")

    ratio_input = gr.Radio(
        choices=["10%", "20%", "30%"],
        value="20%",
        label="Final Summary Length",
        info="This controls only the Large BART summary.",
    )

    large_box = gr.Textbox(lines=8, label="Large BART Summary")

    # Store all generated large summaries
    large_10_state = gr.State("")
    large_20_state = gr.State("")
    large_30_state = gr.State("")

    gr.Markdown("---")
    gr.Markdown("## Comparison (Baseline Models)")

    small_no_chunk_box = gr.Textbox(
        label="Small BART — No Chunk (Baseline)",
        lines=6,
    )

    small_chunk_box = gr.Textbox(
        label="Small BART + Chunking",
        lines=8,
    )

    run_btn.click(
        fn=process_video,
        inputs=[video_input],
        outputs=[
            transcript_box,
            large_box,
            small_no_chunk_box,
            small_chunk_box,
            large_10_state,
            large_20_state,
            large_30_state,
        ],
    )

    ratio_input.change(
        fn=select_large_summary,
        inputs=[ratio_input, large_10_state, large_20_state, large_30_state],
        outputs=large_box,
    )

if __name__ == "__main__":
    demo.launch()
