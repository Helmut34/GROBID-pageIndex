"""
Granite4 / Ollama utilities — drop-in replacements for the OpenAI helpers in utils.py.

Provides sync and async chat completions via Ollama so the page index
pipeline can run against a local Granite4 model without an API key.
"""

import asyncio
import logging
import time

import ollama

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "granite4"


# ------------------------------------------------------------------
# Token counting (rough estimate — Ollama doesn't expose a tokenizer)
# ------------------------------------------------------------------

def count_tokens(text: str, model: str = DEFAULT_MODEL) -> int:
    """
    Approximate token count.  Granite uses a SentencePiece-style tokenizer;
    a rough heuristic of ~3.5 chars/token works for planning purposes.
    """
    if not text:
        return 0
    return max(1, int(len(text) / 3.5))


# ------------------------------------------------------------------
# Synchronous chat
# ------------------------------------------------------------------

def granite_chat(
    prompt: str,
    model: str = DEFAULT_MODEL,
    chat_history: list | None = None,
    temperature: float = 0.0,
    max_retries: int = 5,
) -> str:
    messages = list(chat_history) if chat_history else []
    messages.append({"role": "user", "content": prompt})

    for attempt in range(max_retries):
        try:
            resp = ollama.chat(
                model=model,
                messages=messages,
                options={"temperature": temperature},
            )
            return resp.message.content
        except Exception as e:
            logger.warning(f"Ollama attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                logger.error("Max retries reached")
                return ""


# ------------------------------------------------------------------
# Async chat
# ------------------------------------------------------------------

async def granite_chat_async(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_retries: int = 5,
) -> str:
    messages = [{"role": "user", "content": prompt}]

    for attempt in range(max_retries):
        try:
            client = ollama.AsyncClient()
            resp = await client.chat(
                model=model,
                messages=messages,
                options={"temperature": temperature},
            )
            return resp.message.content
        except Exception as e:
            logger.warning(f"Ollama async attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
            else:
                logger.error("Max retries reached")
                return ""


# ------------------------------------------------------------------
# Node summary (matches interface in utils.py)
# ------------------------------------------------------------------

async def generate_node_summary(node: dict, model: str = DEFAULT_MODEL) -> str:
    prompt = (
        "You are given a part of a document. Generate a brief description of "
        "the main points covered in this section.\n\n"
        f"Section Text: {node['text']}\n\n"
        "Return only the description, no preamble."
    )
    return await granite_chat_async(prompt, model=model)


async def generate_summaries_for_structure(structure, model: str = DEFAULT_MODEL):
    """Generate summaries for all nodes in a tree structure."""
    from .utils import structure_to_list

    nodes = structure_to_list(structure)
    tasks = [generate_node_summary(node, model=model) for node in nodes]
    summaries = await asyncio.gather(*tasks)

    for node, summary in zip(nodes, summaries):
        node["summary"] = summary
    return structure


def generate_doc_description(structure, model: str = DEFAULT_MODEL) -> str:
    prompt = (
        "You are an expert at summarising academic documents.\n"
        "Given this document structure, generate a one-sentence description "
        "that captures the paper's main contribution.\n\n"
        f"Document Structure: {structure}\n\n"
        "Return only the description."
    )
    return granite_chat(prompt, model=model)
