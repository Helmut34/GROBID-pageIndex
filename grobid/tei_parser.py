"""
Parse GROBID TEI XML into structured Python dicts.

Extracts:
  - Paper metadata (title, authors, abstract, date)
  - Sections / subsections with full text
  - Inline citation references (linked to bibliography)
  - Bibliography entries
"""

from lxml import etree

TEI_NS = "http://www.tei-c.org/ns/1.0"
NS = {"tei": TEI_NS}


def _text_of(elem, default: str = "") -> str:
    """Get all text content of an element, stripping tags."""
    if elem is None:
        return default
    return "".join(elem.itertext()).strip()


def _attr(elem, attr: str, default: str = "") -> str:
    if elem is None:
        return default
    return (elem.get(attr) or default).strip()


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def parse_tei(tei_xml: str) -> dict:
    """
    Parse full GROBID TEI XML into a structured dict.

    Returns:
        {
            "title": str,
            "authors": [{"name": str, "affiliation": str}, ...],
            "abstract": str,
            "date": str,
            "sections": [
                {
                    "heading": str,
                    "section_num": str,
                    "level": int,          # 1 for top-level, 2 for sub, etc.
                    "text": str,           # full text of the section body
                    "citations": [         # inline citation keys used here
                        {"key": str, "text": str}
                    ]
                }, ...
            ],
            "bibliography": {
                "#b0": {
                    "title": str,
                    "authors": [str],
                    "date": str,
                    "journal": str,
                    "doi": str
                }, ...
            }
        }
    """
    root = etree.fromstring(tei_xml.encode("utf-8"))

    result = {}
    result["title"] = _parse_title(root)
    result["authors"] = _parse_authors(root)
    result["abstract"] = _parse_abstract(root)
    result["date"] = _parse_date(root)
    result["sections"] = _parse_body_sections(root)
    result["bibliography"] = _parse_bibliography(root)

    return result


# ------------------------------------------------------------------
# Internal parsers
# ------------------------------------------------------------------

def _parse_title(root) -> str:
    title_elem = root.find(".//tei:titleStmt/tei:title[@type='main']", NS)
    if title_elem is None:
        title_elem = root.find(".//tei:titleStmt/tei:title", NS)
    return _text_of(title_elem)


def _parse_authors(root) -> list:
    authors = []
    for author_elem in root.findall(
        ".//tei:sourceDesc//tei:author", NS
    ):
        persname = author_elem.find("tei:persName", NS)
        if persname is None:
            continue
        first = _text_of(persname.find("tei:forename", NS))
        last = _text_of(persname.find("tei:surname", NS))
        name = f"{first} {last}".strip()

        aff_elem = author_elem.find("tei:affiliation", NS)
        affiliation = ""
        if aff_elem is not None:
            org = aff_elem.find("tei:orgName[@type='institution']", NS)
            affiliation = _text_of(org)

        authors.append({"name": name, "affiliation": affiliation})
    return authors


def _parse_abstract(root) -> str:
    abstract_elem = root.find(".//tei:profileDesc/tei:abstract", NS)
    return _text_of(abstract_elem)


def _parse_date(root) -> str:
    date_elem = root.find(
        ".//tei:sourceDesc//tei:date[@type='published']", NS
    )
    if date_elem is None:
        date_elem = root.find(".//tei:sourceDesc//tei:date", NS)
    return _attr(date_elem, "when", _text_of(date_elem))


def _parse_body_sections(root) -> list:
    """Walk <body> divs. GROBID nests them as <div> with <head> children."""
    body = root.find(".//tei:body", NS)
    if body is None:
        return []

    sections = []
    _walk_divs(body, sections, level=1)
    return sections


def _walk_divs(parent, sections: list, level: int):
    for div in parent.findall("tei:div", NS):
        head = div.find("tei:head", NS)
        heading = _text_of(head) if head is not None else ""

        # Extract the `n` attribute from <head> for section numbering
        section_num = _attr(head, "n") if head is not None else ""

        paragraphs = []
        citations = []
        for p in div.findall("tei:p", NS):
            p_text, p_cites = _extract_paragraph(p)
            paragraphs.append(p_text)
            citations.extend(p_cites)

        text = "\n\n".join(paragraphs)

        sections.append({
            "heading": heading,
            "section_num": section_num,
            "level": level,
            "text": text,
            "citations": citations,
        })

        # Recurse into nested divs
        _walk_divs(div, sections, level=level + 1)


def _extract_paragraph(p_elem) -> tuple:
    """
    Extract text and inline citation refs from a <p> element.
    Returns (text_str, [{"key": "#b0", "text": "Author et al."}]).
    """
    citations = []
    parts = []

    if p_elem.text:
        parts.append(p_elem.text)

    for child in p_elem:
        tag = etree.QName(child.tag).localname if child.tag else ""

        if tag == "ref":
            ref_type = child.get("type", "")
            target = child.get("target", "")
            ref_text = _text_of(child)

            if ref_type == "bibr" and target:
                citations.append({"key": target, "text": ref_text})
                parts.append(f"[{ref_text}]")
            else:
                parts.append(ref_text)
        else:
            parts.append(_text_of(child))

        if child.tail:
            parts.append(child.tail)

    return "".join(parts), citations


def _parse_bibliography(root) -> dict:
    """Parse <listBibl> into a dict keyed by xml:id → citation metadata."""
    bibl = {}
    for entry in root.findall(
        ".//tei:listBibl/tei:biblStruct", NS
    ):
        xml_id = entry.get("{http://www.w3.org/XML/1998/namespace}id", "")
        key = f"#{xml_id}" if xml_id else ""

        # Title
        title_elem = entry.find(
            ".//tei:analytic/tei:title[@type='main']", NS
        )
        if title_elem is None:
            title_elem = entry.find(".//tei:monogr/tei:title", NS)
        title = _text_of(title_elem)

        # Authors
        authors = []
        for author_elem in entry.findall(".//tei:analytic//tei:author/tei:persName", NS):
            first = _text_of(author_elem.find("tei:forename", NS))
            last = _text_of(author_elem.find("tei:surname", NS))
            authors.append(f"{first} {last}".strip())
        if not authors:
            for author_elem in entry.findall(".//tei:monogr//tei:author/tei:persName", NS):
                first = _text_of(author_elem.find("tei:forename", NS))
                last = _text_of(author_elem.find("tei:surname", NS))
                authors.append(f"{first} {last}".strip())

        # Date
        date_elem = entry.find(".//tei:monogr/tei:imprint/tei:date", NS)
        date = _attr(date_elem, "when", _text_of(date_elem))

        # Journal
        journal_elem = entry.find(".//tei:monogr/tei:title[@level='j']", NS)
        journal = _text_of(journal_elem)

        # DOI
        doi_elem = entry.find(".//tei:idno[@type='DOI']", NS)
        doi = _text_of(doi_elem)

        if key:
            bibl[key] = {
                "title": title,
                "authors": authors,
                "date": date,
                "journal": journal,
                "doi": doi,
            }

    return bibl
