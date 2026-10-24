
import fitz
import re
from pathlib import Path


def extract_text_from_pdf(pdf_path):
    pages = []

    with fitz.open(pdf_path) as doc:
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text("text")

            if text.strip():
                # Normalize whitespace without changing the wording.
                text = re.sub(r"\s+", " ", text).strip()

                pages.append({
                    "page": page_number,
                    "text": text
                })

    return pages


def chunk_text(pages, chunk_size=1000, overlap=200):
    chunks = []

    for page in pages:
        text = page["text"].strip()

        if not text:
            continue

        # Split at sentence boundaries instead of arbitrary positions.
        sentences = re.split(r"(?<=[.!?])\s+", text)

        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue

            # Handle unusually long sentences.
            if len(sentence) > chunk_size:
                if current_chunk:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "page": page["page"]
                    })
                    current_chunk = ""

                start = 0

                while start < len(sentence):
                    piece = sentence[start:start + chunk_size].strip()

                    if piece:
                        chunks.append({
                            "text": piece,
                            "page": page["page"]
                        })

                    if start + chunk_size >= len(sentence):
                        break

                    start += chunk_size - overlap

                continue

            # Add the sentence if it fits in the current chunk.
            if not current_chunk:
                current_chunk = sentence

            elif len(current_chunk) + len(sentence) + 1 <= chunk_size:
                current_chunk += " " + sentence

            else:
                chunks.append({
                    "text": current_chunk.strip(),
                    "page": page["page"]
                })

                # Preserve some previous context between chunks.
                previous_sentences = current_chunk.split(". ")
                overlap_text = ""

                for previous in reversed(previous_sentences):
                    candidate = (
                        previous if not overlap_text
                        else previous + ". " + overlap_text
                    )

                    if len(candidate) > overlap:
                        break

                    overlap_text = candidate

                if overlap_text:
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence

        # Save the final chunk from this page.
        if current_chunk.strip():
            chunks.append({
                "text": current_chunk.strip(),
                "page": page["page"]
            })

    return chunks


def process_pdf(pdf_path):
    pages = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(pages)

    return chunks


if __name__ == "__main__":
    pdf_path = input(
        "Enter the path to your PDF: "
    ).strip().strip('"')

    if not Path(pdf_path).is_file():
        print("Error: PDF file not found.")
    else:
        chunks = process_pdf(pdf_path)

        print("\nPDF processed successfully!")
        print(f"Total chunks: {len(chunks)}")

        if chunks:
            print("\nFirst chunk:")
            print(chunks[0]["text"][:500])
            print(f"\nPage number: {chunks[0]['page']}")
