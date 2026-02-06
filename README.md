Preliminary pipeline for project: parses academic PDFs with simulated GROBID, builds a hierarchical page index, and uses Granite4 (via Ollama) for RAG-based paper summarisation, citation analysis, and question answering.

## Pipeline

```
PDF → GROBID (TEI XML) → TEI Parser → Markdown → Page Index Tree → Granite4 RAG
```

| Step | Module | What it does |
|------|--------|--------------|
| 1 | `grobid/client.py` | Sends PDF to GROBID REST API, returns TEI XML |
| 2 | `grobid/tei_parser.py` | Extracts title, authors, sections, inline citations, bibliography |
| 3 | `grobid/tei_to_markdown.py` | Converts parsed TEI into heading-structured markdown |
| 4 | `rag/page_index_md.py` | Builds hierarchical tree index from markdown headings |
| 5 | `pipeline.py` | Orchestrates all steps; queries Granite4 for summaries and answers |

## Requirements

- **GROBID** service running (Docker): `docker run --rm -p 8070:8070 lfoppiano/grobid:0.8.1`
- **Granite4** via Ollama: `ollama pull granite4`
- Python 3.10+: `pip install -r requirements.txt`

## Usage

```bash
# Full pipeline
python pipeline.py paper.pdf --summary --query "What is the main contribution?"

# Run tests (no GROBID needed — uses sample TEI XML)
python test/test_pipeline.py
```

## Why PageIndex RAG for Scholarly Applications

Traditional RAG approaches chunk documents into flat, fixed-size text blocks — losing the structural relationships that make academic papers meaningful. PageIndex RAG addresses this by preserving the paper's hierarchy as a tree:

- **Structure-aware retrieval.** A research paper is not a bag of paragraphs. Sections nest under sections, arguments build across headings, and conclusions reference methodology. The page index tree keeps these relationships intact, so Granite4 can reason about *where* information sits in the paper's argument — not just *what* words appear nearby.

- **Citation-grounded answers.** GROBID extracts inline citation markers and links them to bibliography entries. The page index carries these through, so when the model summarises a section or answers a query, it can trace claims back to the specific references the authors cited. This is critical for literature review workflows where provenance matters.

- **Efficient context usage.** Instead of stuffing an entire paper into the LLM context, the tree structure lets the pipeline send targeted subtrees — a section and its children — keeping token usage low while retaining the surrounding context (parent headings, sibling sections). This matters especially for smaller local models like Granite4 with limited context windows.

- **Scalable to multi-paper workflows.** Each paper produces a self-contained page index with node IDs and summaries. Multiple paper indexes can be combined for cross-paper queries ("How does paper A's methodology compare to paper B?") without re-processing the PDFs — the tree structures are lightweight JSON.

- **Reproducible structure.** GROBID's TEI parsing is deterministic for a given PDF. The page index tree built from it is stable across runs, making results reproducible — an important property for scholarly tools where users expect consistent outputs.

## Credits

- **PageIndex RAG** — by [Vectify AI](https://github.com/Vectify) (MIT License, 2025). The `rag/page_index_md.py` and `rag/utils.py` modules originate from their work on hierarchical document indexing for retrieval-augmented generation.

- **GROBID** — by Patrice Lopez and contributors. A machine learning library for extracting structured information from scholarly PDFs into TEI XML. [github.com/kermitt2/grobid](https://github.com/kermitt2/grobid) (Apache 2.0 License).

## Status

This is a preliminary setup. The TEI parsing, page index, and Granite4 RAG stages are functional and tested. The GROBID step requires Docker or Java 21 to run the service.
