"""
simplyResearch Pipeline: PDF → GROBID → PageIndex → Granite4 RAG

Usage:
    python pipeline.py <pdf_path> [--grobid-url http://localhost:8070]
                                  [--model granite4]
                                  [--summary]
                                  [--query "your question here"]
"""

import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

# Local imports
sys.path.insert(0, str(Path(__file__).parent))

from grobid.client import process_pdf, is_alive, DEFAULT_GROBID_URL
from grobid.tei_parser import parse_tei
from grobid.tei_to_markdown import parsed_tei_to_markdown
from rag.page_index_md import md_to_tree
from rag.granite_utils import (
    granite_chat,
    generate_doc_description,
    count_tokens,
    DEFAULT_MODEL,
)


def step_grobid_parse(pdf_path: str, grobid_url: str) -> dict:
    """Step 1: Send PDF to GROBID, get parsed TEI structure."""
    print(f"[1/4] Parsing PDF with GROBID...")

    if not is_alive(grobid_url):
        print(
            f"  ERROR: GROBID is not running at {grobid_url}\n"
            f"  Start it with: docker run --rm -p 8070:8070 lfoppiano/grobid:0.8.1"
        )
        sys.exit(1)

    tei_xml = process_pdf(pdf_path, grobid_url=grobid_url)
    parsed = parse_tei(tei_xml)

    print(f"  Title: {parsed['title']}")
    print(f"  Authors: {len(parsed['authors'])}")
    print(f"  Sections: {len(parsed['sections'])}")
    print(f"  Bibliography entries: {len(parsed['bibliography'])}")

    return parsed


def step_to_markdown(parsed: dict, output_dir: str) -> str:
    """Step 2: Convert parsed TEI to markdown file for page index."""
    print(f"[2/4] Converting to structured markdown...")

    markdown = parsed_tei_to_markdown(parsed)
    md_path = os.path.join(output_dir, "paper.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    token_count = count_tokens(markdown)
    print(f"  Markdown written: {md_path} (~{token_count} tokens)")

    return md_path


async def step_build_page_index(md_path: str, model: str, add_summary: bool) -> dict:
    """Step 3: Build page index tree from the markdown."""
    print(f"[3/4] Building page index tree...")

    tree = await md_to_tree(
        md_path=md_path,
        if_thinning=False,
        if_add_node_summary="yes" if add_summary else "no",
        summary_token_threshold=200,
        model=model,
        if_add_node_text="yes",
        if_add_node_id="yes",
    )

    structure = tree.get("structure", [])
    node_count = _count_nodes(structure)
    print(f"  Page index built: {node_count} nodes")

    return tree


def step_granite_rag(
    page_index: dict,
    parsed_tei: dict,
    model: str,
    query: str | None = None,
) -> dict:
    """Step 4: Use Granite4 with page index context for summarisation & queries."""
    print(f"[4/4] Running Granite4 RAG...")

    results = {}

    # --- Paper summary ---
    summary_prompt = _build_summary_prompt(page_index, parsed_tei)
    print("  Generating paper summary...")
    results["summary"] = granite_chat(summary_prompt, model=model)

    # --- Citation analysis ---
    if parsed_tei.get("bibliography"):
        citation_prompt = _build_citation_prompt(page_index, parsed_tei)
        print("  Analysing citations...")
        results["citation_analysis"] = granite_chat(citation_prompt, model=model)

    # --- User query ---
    if query:
        query_prompt = _build_query_prompt(page_index, parsed_tei, query)
        print(f"  Answering query: {query}")
        results["query_answer"] = granite_chat(query_prompt, model=model)

    return results


# ------------------------------------------------------------------
# Prompt builders
# ------------------------------------------------------------------

def _build_summary_prompt(page_index: dict, parsed: dict) -> str:
    index_json = json.dumps(page_index, indent=2, ensure_ascii=False, default=str)
    # Truncate if too long for context
    if len(index_json) > 12000:
        index_json = index_json[:12000] + "\n... [truncated]"

    return (
        "You are an academic research assistant. "
        "Given the following page index of a research paper, write a brief summary "
        "(3-5 sentences) that captures the paper's main contribution, methodology, "
        "and key findings.\n\n"
        f"Paper Title: {parsed.get('title', 'Unknown')}\n\n"
        f"Page Index:\n{index_json}\n\n"
        "Summary:"
    )


def _build_citation_prompt(page_index: dict, parsed: dict) -> str:
    # Build a compact citations view
    bib = parsed.get("bibliography", {})
    bib_summary = []
    for key, entry in list(bib.items())[:30]:  # Limit to 30 for context
        authors = ", ".join(entry.get("authors", [])[:3])
        title = entry.get("title", "")
        bib_summary.append(f"  {key}: {authors}. \"{title}\" ({entry.get('date', '')})")
    bib_text = "\n".join(bib_summary)

    # Collect all inline citation usages from sections
    section_cites = []
    for section in parsed.get("sections", []):
        if section.get("citations"):
            heading = section.get("heading", "Unnamed")
            cite_keys = [c["key"] for c in section["citations"]]
            section_cites.append(f"  Section \"{heading}\": {cite_keys}")
    cite_text = "\n".join(section_cites[:20])

    return (
        "You are an academic research assistant. "
        "Analyse how the citations in this paper support its arguments.\n\n"
        f"Paper Title: {parsed.get('title', 'Unknown')}\n\n"
        f"Bibliography:\n{bib_text}\n\n"
        f"Citation usage by section:\n{cite_text}\n\n"
        "Provide a brief analysis of:\n"
        "1. Which citations are most central to the paper's argument\n"
        "2. How different sections rely on different citation groups\n"
        "3. Any patterns in the citation usage\n\n"
        "Analysis:"
    )


def _build_query_prompt(page_index: dict, parsed: dict, query: str) -> str:
    index_json = json.dumps(page_index, indent=2, ensure_ascii=False, default=str)
    if len(index_json) > 12000:
        index_json = index_json[:12000] + "\n... [truncated]"

    return (
        "You are an academic research assistant. "
        "Answer the following question based on the paper's page index.\n\n"
        f"Paper Title: {parsed.get('title', 'Unknown')}\n\n"
        f"Page Index:\n{index_json}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _count_nodes(structure) -> int:
    count = 0
    if isinstance(structure, list):
        for item in structure:
            count += _count_nodes(item)
    elif isinstance(structure, dict):
        count += 1
        if "nodes" in structure:
            count += _count_nodes(structure["nodes"])
    return count


async def run_pipeline(
    pdf_path: str,
    grobid_url: str = DEFAULT_GROBID_URL,
    model: str = DEFAULT_MODEL,
    add_summary: bool = True,
    query: str | None = None,
    output_dir: str | None = None,
):
    """Run the full pipeline and return results."""

    if output_dir is None:
        pdf_name = Path(pdf_path).stem
        output_dir = os.path.join(Path(__file__).parent, "results", pdf_name)
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: GROBID
    parsed_tei = step_grobid_parse(pdf_path, grobid_url)

    # Save raw parsed TEI
    with open(os.path.join(output_dir, "parsed_tei.json"), "w") as f:
        json.dump(parsed_tei, f, indent=2, ensure_ascii=False, default=str)

    # Step 2: Markdown
    md_path = step_to_markdown(parsed_tei, output_dir)

    # Step 3: Page Index
    page_index = await step_build_page_index(md_path, model, add_summary)

    # Save page index
    with open(os.path.join(output_dir, "page_index.json"), "w") as f:
        json.dump(page_index, f, indent=2, ensure_ascii=False, default=str)

    # Step 4: Granite4 RAG
    rag_results = step_granite_rag(page_index, parsed_tei, model, query)

    # Save results
    with open(os.path.join(output_dir, "rag_results.json"), "w") as f:
        json.dump(rag_results, f, indent=2, ensure_ascii=False)

    print(f"\nAll outputs saved to: {output_dir}/")
    return {
        "parsed_tei": parsed_tei,
        "page_index": page_index,
        "rag_results": rag_results,
    }


def main():
    parser = argparse.ArgumentParser(description="simplyResearch: PDF → GROBID → PageIndex → Granite4")
    parser.add_argument("pdf_path", help="Path to the PDF file to process")
    parser.add_argument("--grobid-url", default=DEFAULT_GROBID_URL, help="GROBID service URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--summary", action="store_true", help="Generate node summaries")
    parser.add_argument("--query", type=str, default=None, help="Question to answer about the paper")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    results = asyncio.run(
        run_pipeline(
            pdf_path=args.pdf_path,
            grobid_url=args.grobid_url,
            model=args.model,
            add_summary=args.summary,
            query=args.query,
            output_dir=args.output_dir,
        )
    )

    # Print summary
    rag = results["rag_results"]
    print("\n" + "=" * 60)
    if rag.get("summary"):
        print("PAPER SUMMARY:")
        print(rag["summary"])
    if rag.get("citation_analysis"):
        print("\nCITATION ANALYSIS:")
        print(rag["citation_analysis"])
    if rag.get("query_answer"):
        print(f"\nQUERY ANSWER:")
        print(rag["query_answer"])
    print("=" * 60)


if __name__ == "__main__":
    main()
