import asyncio
import logging
from dataclasses import dataclass
from typing import AsyncIterator

from stream2sentence import generate_sentences_async


# -----------------------------------------------------------------------------
# Text Extraction
# -----------------------------------------------------------------------------


async def text_stream_generator(text_stream: AsyncIterator) -> AsyncIterator[str]:

    async for chunk in text_stream:
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content


# -----------------------------------------------------------------------------
# Sentence Generation
# -----------------------------------------------------------------------------


async def generate_sentences_from_stream(text_stream: AsyncIterator, min_first_fragment_length: int = 10, min_sentence_length: int = 20) -> AsyncIterator[str]:

    text_stream = text_stream_generator(text_stream)

    async for sentence in generate_sentences_async(
        text_stream,
        minimum_first_fragment_length=min_first_fragment_length,
        minimum_sentence_length=min_sentence_length,
    ):
        sentence = sentence.strip()
        if sentence:
            yield sentence