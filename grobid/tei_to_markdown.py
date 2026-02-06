"""
Convert parsed GROBID TEI (from tei_parser) into structured markdown.

The markdown output uses heading levels that match the page_index_md.py system,
so the existing tree builder can be reused directly.

Citations are preserved inline as [AuthorYear] and a ## References section
is appended at the end with full bibliography entries.
"""


def parsed_tei_to_markdown(parsed: dict) -> str:
    """
    Convert the dict produced by tei_parser.parse_tei() into a markdown string.

    Structure:
        # <Title>
        **Authors:** ...
        **Date:** ...

        ## Abstract
        ...

        ## <Section Heading>
        <body text with [citation] markers>

        ## References
        [1] Author et al. "Title". Journal, Year. DOI
    """
    lines = []

    # Title
    title = parsed.get("title", "Untitled")
    lines.append(f"# {title}")
    lines.append("")

    # Metadata
    authors = parsed.get("authors", [])
    if authors:
        author_str = "; ".join(
            a["name"] + (f" ({a['affiliation']})" if a.get("affiliation") else "")
            for a in authors
        )
        lines.append(f"**Authors:** {author_str}")
        lines.append("")

    date = parsed.get("date", "")
    if date:
        lines.append(f"**Date:** {date}")
        lines.append("")

    # Abstract
    abstract = parsed.get("abstract", "")
    if abstract:
        lines.append("## Abstract")
        lines.append("")
        lines.append(abstract)
        lines.append("")

    # Body sections
    bibliography = parsed.get("bibliography", {})
    bib_index = _build_bib_index(bibliography)

    for section in parsed.get("sections", []):
        heading = section.get("heading", "")
        level = section.get("level", 1)
        section_num = section.get("section_num", "")
        text = section.get("text", "")

        # Build heading line: ## 1.2 Introduction
        hashes = "#" * (level + 1)  # level 1 → ##, level 2 → ###, etc.
        heading_parts = []
        if section_num:
            heading_parts.append(section_num)
        if heading:
            heading_parts.append(heading)
        heading_str = " ".join(heading_parts) if heading_parts else "Untitled Section"

        lines.append(f"{hashes} {heading_str}")
        lines.append("")

        if text:
            lines.append(text)
            lines.append("")

    # Bibliography / References
    if bibliography:
        lines.append("## References")
        lines.append("")
        for key, entry in bibliography.items():
            idx = bib_index.get(key, key)
            ref_line = _format_bib_entry(idx, entry)
            lines.append(ref_line)
            lines.append("")

    return "\n".join(lines)


def _build_bib_index(bibliography: dict) -> dict:
    """Map bibliography keys (#b0, #b1, ...) to sequential numbers."""
    index = {}
    for i, key in enumerate(bibliography.keys(), start=1):
        index[key] = i
    return index


def _format_bib_entry(idx, entry: dict) -> str:
    """Format a single bibliography entry as a markdown line."""
    parts = [f"[{idx}]"]

    authors = entry.get("authors", [])
    if authors:
        parts.append(", ".join(authors) + ".")

    title = entry.get("title", "")
    if title:
        parts.append(f'"{title}".')

    journal = entry.get("journal", "")
    if journal:
        parts.append(f"*{journal}*.")

    date = entry.get("date", "")
    if date:
        parts.append(f"({date}).")

    doi = entry.get("doi", "")
    if doi:
        parts.append(f"DOI: {doi}")

    return " ".join(parts)
