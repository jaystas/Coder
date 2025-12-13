"""
Low-latency sentence streaming pipeline for text-to-speech.

This module provides async functions to process text streams (e.g., from AsyncOpenAI)
into sentences using a producer/consumer pattern with asyncio queues.

Architecture:
    AsyncOpenAI Stream → Text Extractor → Sentence Generator → Queue → TTS Consumer

Usage:
    # Option 1: Direct async generator (simpler, single coroutine)
    async for sentence in generate_sentences_from_stream(openai_stream):
        audio = await tts_generate(sentence)
        await play(audio)

    # Option 2: Producer/consumer with queue (concurrent, lower latency)
    queue: asyncio.Queue[SentenceQueueItem] = asyncio.Queue()

    async with asyncio.TaskGroup() as tg:
        tg.create_task(stream_sentences_to_queue(openai_stream, queue))
        tg.create_task(tts_consumer(queue))
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import AsyncIterator

from stream2sentence import generate_sentences_async

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Data Types
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SentenceItem:
    """A sentence ready for TTS processing."""

    text: str
    index: int


@dataclass(frozen=True, slots=True)
class StreamComplete:
    """Sentinel indicating the text stream has ended successfully."""

    total_sentences: int


@dataclass(frozen=True, slots=True)
class StreamError:
    """Wraps an exception that occurred during streaming."""

    exception: Exception


# Union type for queue items
SentenceQueueItem = SentenceItem | StreamComplete | StreamError


# -----------------------------------------------------------------------------
# Text Extraction
# -----------------------------------------------------------------------------


async def extract_text_deltas(openai_stream: AsyncIterator) -> AsyncIterator[str]:
    """
    Extract text content from an AsyncOpenAI chat completion stream.

    Args:
        openai_stream: Async iterator from AsyncOpenAI client.
            Expected structure: chunk.choices[0].delta.content

    Yields:
        Text deltas (strings) as they arrive from the stream.
    """
    async for chunk in openai_stream:
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content


# -----------------------------------------------------------------------------
# Sentence Generation (Direct Async Generator)
# -----------------------------------------------------------------------------


async def generate_sentences_from_stream(
    openai_stream: AsyncIterator,
    *,
    quick_yield: bool = True,
    min_first_fragment_length: int = 10,
    min_sentence_length: int = 10,
) -> AsyncIterator[str]:
    """
    Async generator that yields sentences from an OpenAI stream.

    Use this for simpler cases where you don't need a queue-based
    producer/consumer pattern. Sentences are yielded as soon as
    they're detected.

    Args:
        openai_stream: Async iterator from AsyncOpenAI chat completions.
        quick_yield: If True, yields sentence fragments quickly for
            lower time-to-first-audio. Recommended for TTS.
        min_first_fragment_length: Minimum characters before yielding
            the first sentence fragment.
        min_sentence_length: Minimum sentence length in characters.

    Yields:
        Sentences as strings, ready for TTS processing.

    Example:
        async for sentence in generate_sentences_from_stream(response):
            audio = await tts.synthesize(sentence)
            await audio_queue.put(audio)
    """
    text_stream = extract_text_deltas(openai_stream)

    async for sentence in generate_sentences_async(
        text_stream,
        quick_yield_single_sentence_fragment=quick_yield,
        quick_yield_for_all_sentences=quick_yield,
        minimum_first_fragment_length=min_first_fragment_length,
        minimum_sentence_length=min_sentence_length,
        cleanup_text_emojis=True,
    ):
        text = sentence.strip()
        if text:
            yield text


# -----------------------------------------------------------------------------
# Queue-Based Producer (for Concurrent Processing)
# -----------------------------------------------------------------------------


async def produce_sentences_to_queue(
    text_stream: AsyncIterator[str],
    sentence_queue: asyncio.Queue[SentenceQueueItem],
    *,
    quick_yield: bool = True,
    min_first_fragment_length: int = 10,
    min_sentence_length: int = 10,
) -> int:
    """
    Generate sentences from a text stream and queue them for TTS.

    This is the core producer function. It consumes text chunks,
    uses stream2sentence for sentence boundary detection, and
    puts SentenceItems onto the queue for downstream processing.

    Args:
        text_stream: Async iterator yielding text chunks.
        sentence_queue: Queue to put sentence items onto.
        quick_yield: Yield fragments quickly for lower latency.
        min_first_fragment_length: Minimum chars for first fragment.
        min_sentence_length: Minimum sentence length.

    Returns:
        Total number of sentences produced.

    Note:
        Always puts a StreamComplete or StreamError as the final item.
        Consumers should check for these to know when processing is done.
    """
    count = 0

    try:
        async for sentence in generate_sentences_async(
            text_stream,
            quick_yield_single_sentence_fragment=quick_yield,
            quick_yield_for_all_sentences=quick_yield,
            minimum_first_fragment_length=min_first_fragment_length,
            minimum_sentence_length=min_sentence_length,
            cleanup_text_emojis=True,
        ):
            text = sentence.strip()
            if text:
                item = SentenceItem(text=text, index=count)
                await sentence_queue.put(item)
                logger.debug("Queued sentence %d: %.50s...", count, text)
                count += 1

    except Exception as e:
        logger.exception("Error in sentence producer")
        await sentence_queue.put(StreamError(exception=e))
        raise

    finally:
        await sentence_queue.put(StreamComplete(total_sentences=count))
        logger.debug("Sentence producer complete: %d sentences", count)

    return count


async def stream_sentences_to_queue(
    openai_stream: AsyncIterator,
    sentence_queue: asyncio.Queue[SentenceQueueItem],
    *,
    quick_yield: bool = True,
    min_first_fragment_length: int = 10,
    min_sentence_length: int = 10,
) -> int:
    """
    Process an OpenAI stream into queued sentences for TTS.

    This is the main entry point for the producer/consumer pattern.
    It extracts text from the OpenAI stream, generates sentences,
    and queues them. Run this as an asyncio task alongside your
    TTS consumer for concurrent processing.

    Args:
        openai_stream: Async iterator from AsyncOpenAI chat completions.
        sentence_queue: Queue where SentenceItems will be placed.
        quick_yield: Yield fragments quickly for lower latency.
        min_first_fragment_length: Minimum chars for first fragment.
        min_sentence_length: Minimum sentence length.

    Returns:
        Total number of sentences produced.

    Example:
        queue: asyncio.Queue[SentenceQueueItem] = asyncio.Queue()

        async def tts_consumer(q: asyncio.Queue[SentenceQueueItem]):
            while True:
                item = await q.get()
                if isinstance(item, StreamComplete):
                    break
                if isinstance(item, StreamError):
                    raise item.exception
                # item is SentenceItem
                audio = await tts.synthesize(item.text)
                await play(audio)

        async with asyncio.TaskGroup() as tg:
            tg.create_task(stream_sentences_to_queue(response, queue))
            tg.create_task(tts_consumer(queue))
    """
    text_stream = extract_text_deltas(openai_stream)
    return await produce_sentences_to_queue(
        text_stream,
        sentence_queue,
        quick_yield=quick_yield,
        min_first_fragment_length=min_first_fragment_length,
        min_sentence_length=min_sentence_length,
    )
