"""
Tests for the GROBID → PageIndex → Granite4 pipeline.

Run:  python -m pytest test/test_pipeline.py -v
  or: python test/test_pipeline.py          (standalone)

Tests that need a live GROBID server are marked and skipped automatically.
"""

import asyncio
import json
import os
import sys
import tempfile

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, PROJECT_ROOT)
# Also add rag/ so the fallback bare import in page_index_md.py works
sys.path.insert(0, os.path.join(PROJECT_ROOT, "rag"))

from grobid.tei_parser import parse_tei
from grobid.tei_to_markdown import parsed_tei_to_markdown
from grobid.client import is_alive
from rag.page_index_md import md_to_tree
from rag.granite_utils import count_tokens, granite_chat

# --------------------------------------------------------------------------
# Sample TEI XML (representative excerpt from a GROBID-processed paper)
# --------------------------------------------------------------------------
SAMPLE_TEI = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0"
     xmlns:xlink="http://www.w3.org/1999/xlink">
  <teiHeader>
    <fileDesc>
      <titleStmt>
        <title type="main">Attention Is All You Need</title>
      </titleStmt>
      <sourceDesc>
        <biblStruct>
          <analytic>
            <author>
              <persName><forename>Ashish</forename><surname>Vaswani</surname></persName>
              <affiliation><orgName type="institution">Google Brain</orgName></affiliation>
            </author>
            <author>
              <persName><forename>Noam</forename><surname>Shazeer</surname></persName>
              <affiliation><orgName type="institution">Google Brain</orgName></affiliation>
            </author>
          </analytic>
          <monogr>
            <imprint>
              <date type="published" when="2017-06-12"/>
            </imprint>
          </monogr>
        </biblStruct>
      </sourceDesc>
    </fileDesc>
    <profileDesc>
      <abstract>
        <p>The dominant sequence transduction models are based on complex recurrent or
        convolutional neural networks. We propose a new simple network architecture,
        the Transformer, based solely on attention mechanisms.</p>
      </abstract>
    </profileDesc>
  </teiHeader>
  <text>
    <body>
      <div>
        <head n="1">Introduction</head>
        <p>Recurrent neural networks, long short-term memory
        <ref type="bibr" target="#b0">[Hochreiter et al., 1997]</ref> and gated recurrent
        neural networks <ref type="bibr" target="#b1">[Cho et al., 2014]</ref>, in particular,
        have been established as state of the art approaches.</p>
      </div>
      <div>
        <head n="2">Background</head>
        <p>The goal of reducing sequential computation also forms the foundation of
        the Extended Neural GPU <ref type="bibr" target="#b2">[Kaiser &amp; Bengio, 2016]</ref>.</p>
        <div>
          <head n="2.1">Self-Attention</head>
          <p>Self-attention, sometimes called intra-attention, is an attention mechanism
          relating different positions of a single sequence.</p>
        </div>
      </div>
      <div>
        <head n="3">Model Architecture</head>
        <p>Most competitive neural sequence transduction models have an encoder-decoder
        structure <ref type="bibr" target="#b3">[Sutskever et al., 2014]</ref>.</p>
      </div>
    </body>
    <back>
      <listBibl>
        <biblStruct xml:id="b0">
          <analytic>
            <title type="main">Long Short-Term Memory</title>
            <author><persName><forename>Sepp</forename><surname>Hochreiter</surname></persName></author>
            <author><persName><forename>Jürgen</forename><surname>Schmidhuber</surname></persName></author>
          </analytic>
          <monogr>
            <title level="j">Neural Computation</title>
            <imprint><date when="1997"/>
            </imprint>
          </monogr>
        </biblStruct>
        <biblStruct xml:id="b1">
          <analytic>
            <title type="main">Learning Phrase Representations using RNN Encoder-Decoder</title>
            <author><persName><forename>Kyunghyun</forename><surname>Cho</surname></persName></author>
          </analytic>
          <monogr>
            <title level="j">EMNLP</title>
            <imprint><date when="2014"/>
            </imprint>
          </monogr>
        </biblStruct>
        <biblStruct xml:id="b2">
          <analytic>
            <title type="main">Neural GPUs Learn Algorithms</title>
            <author><persName><forename>Lukasz</forename><surname>Kaiser</surname></persName></author>
            <author><persName><forename>Samy</forename><surname>Bengio</surname></persName></author>
          </analytic>
          <monogr>
            <title level="j">ICLR</title>
            <imprint><date when="2016"/>
            </imprint>
          </monogr>
        </biblStruct>
        <biblStruct xml:id="b3">
          <analytic>
            <title type="main">Sequence to Sequence Learning with Neural Networks</title>
            <author><persName><forename>Ilya</forename><surname>Sutskever</surname></persName></author>
          </analytic>
          <monogr>
            <title level="j">NeurIPS</title>
            <imprint><date when="2014"/>
            </imprint>
          </monogr>
        </biblStruct>
      </listBibl>
    </back>
  </text>
</TEI>
"""


def test_tei_parser():
    """Test that parse_tei extracts the correct structure from TEI XML."""
    parsed = parse_tei(SAMPLE_TEI)

    assert parsed["title"] == "Attention Is All You Need"
    assert len(parsed["authors"]) == 2
    assert parsed["authors"][0]["name"] == "Ashish Vaswani"
    assert parsed["authors"][0]["affiliation"] == "Google Brain"
    assert "Transformer" in parsed["abstract"]
    assert parsed["date"] == "2017-06-12"

    # Sections: Introduction, Background, Self-Attention (nested), Model Architecture
    assert len(parsed["sections"]) == 4
    assert parsed["sections"][0]["heading"] == "Introduction"
    assert parsed["sections"][0]["level"] == 1
    assert parsed["sections"][2]["heading"] == "Self-Attention"
    assert parsed["sections"][2]["level"] == 2

    # Citations in Introduction
    intro_cites = parsed["sections"][0]["citations"]
    assert len(intro_cites) == 2
    assert intro_cites[0]["key"] == "#b0"

    # Bibliography
    assert len(parsed["bibliography"]) == 4
    assert parsed["bibliography"]["#b0"]["title"] == "Long Short-Term Memory"
    assert "Hochreiter" in parsed["bibliography"]["#b0"]["authors"][0]

    print("  PASS: tei_parser")


def test_tei_to_markdown():
    """Test that markdown conversion produces valid heading structure."""
    parsed = parse_tei(SAMPLE_TEI)
    md = parsed_tei_to_markdown(parsed)

    assert md.startswith("# Attention Is All You Need")
    assert "## Abstract" in md
    assert "## 1 Introduction" in md
    assert "## 2 Background" in md
    assert "### 2.1 Self-Attention" in md
    assert "## 3 Model Architecture" in md
    assert "## References" in md
    assert "[1]" in md  # first bibliography entry

    print("  PASS: tei_to_markdown")


def test_page_index_from_markdown():
    """Test that the page index tree builder works on GROBID-derived markdown."""
    parsed = parse_tei(SAMPLE_TEI)
    md = parsed_tei_to_markdown(parsed)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(md)
        md_path = f.name

    try:
        tree = asyncio.run(md_to_tree(
            md_path=md_path,
            if_thinning=False,
            if_add_node_summary="no",
            if_add_node_text="yes",
            if_add_node_id="yes",
            model="granite4",
        ))

        assert "doc_name" in tree
        assert "structure" in tree
        structure = tree["structure"]
        assert len(structure) > 0

        # Top-level node should be the paper title
        assert "Attention Is All You Need" in structure[0]["title"]

        print(f"  PASS: page_index ({len(structure)} top-level nodes)")
    finally:
        os.unlink(md_path)


def test_count_tokens():
    """Test that Granite token counting works."""
    assert count_tokens("") == 0
    assert count_tokens("hello world") > 0
    print("  PASS: count_tokens")


def test_granite_chat():
    """Test that Granite4 can respond via Ollama."""
    response = granite_chat("Reply with exactly the word 'OK'.", model="granite4")
    assert len(response) > 0
    print(f"  PASS: granite_chat (response: {response[:50]})")


def test_grobid_alive():
    """Test GROBID connectivity (informational — doesn't fail the suite)."""
    alive = is_alive()
    status = "REACHABLE" if alive else "NOT RUNNING (start with: docker run --rm -p 8070:8070 lfoppiano/grobid:0.8.1)"
    print(f"  INFO: GROBID status: {status}")


if __name__ == "__main__":
    print("=" * 60)
    print("Running pipeline tests...")
    print("=" * 60)

    test_tei_parser()
    test_tei_to_markdown()
    test_page_index_from_markdown()
    test_count_tokens()
    test_granite_chat()
    test_grobid_alive()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
