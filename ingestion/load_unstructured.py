import json
from pathlib import Path

from docx import Document


ROOT = Path(__file__).resolve().parent.parent
HANDBOOK = ROOT / "data" / "raw" / "Hospital_Handbook.docx"
OUTPUT_DIR = ROOT / "data" / "processed" / "chunks"
OUTPUT_FILE = OUTPUT_DIR / "hospital_handbook_chunks.json"

MAX_CHARS = 1200


def read_sections(path):
	"""Read the handbook and group paragraphs under their headings."""
	document = Document(path)
	sections = []
	heading = "General"
	paragraphs = []

	def save_section():
		if paragraphs:
			sections.append((heading, "\n".join(paragraphs)))

	for paragraph in document.paragraphs:
		text = paragraph.text.strip()
		style = paragraph.style.name.lower() if paragraph.style else ""

		if not text or style.startswith("title"):
			continue

		if style.startswith("heading"):
			save_section()
			heading = text
			paragraphs = []
		else:
			paragraphs.append(text)

	save_section()
	return sections


def make_chunks(text):
	"""Split long sections without cutting a paragraph in half."""
	paragraphs = text.split("\n")
	chunks = []
	current = ""

	for paragraph in paragraphs:
		possible = f"{current}\n{paragraph}".strip()

		if current and len(possible) > MAX_CHARS:
			chunks.append(current)
			current = paragraph
		else:
			current = possible

	if current:
		chunks.append(current)

	return chunks


def load_unstructured_data():
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	chunks = []

	for section_name, section_text in read_sections(HANDBOOK):
		for text in make_chunks(section_text):
			chunks.append({
				"chunk_id": f"handbook_{len(chunks):03d}",
				"source": HANDBOOK.name,
				"heading": section_name,
				"text": text,
			})

	with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
		json.dump(chunks, file, indent=2, ensure_ascii=False)

	print(f"Saved {len(chunks)} chunks to {OUTPUT_FILE}")


if __name__ == "__main__":
	load_unstructured_data()
