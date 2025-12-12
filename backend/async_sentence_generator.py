"""
Async Text-to-Sentence Pipeline
-------------------------------
Converts async LLM token streams into sentences using stream2sentence.

Architecture:
    LLM Token Stream (async)
        ↓
    Token Collector (async task → feeds char buffer)
        ↓
    Character Buffer (thread-safe queue.Queue)
        ↓
    Sync Character Iterator (blocking reads)
        ↓
    stream2sentence (runs in thread executor)
        ↓
    Sentence Queue (asyncio.Queue → for TTS)

Usage:
    sentence_queue = asyncio.Queue()

    async for sentence in generate_sentences(llm_stream, sentence_queue):
        print(f"Ready: {sentence}")

    # Or with callback:
    async for sentence in generate_sentences(
        llm_stream,
        sentence_queue,
        on_sentence=lambda s: print(f"Callback: {s}")
    ):
        pass
"""

import asyncio
import logging
import queue
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Optional

import stream2sentence as s2s

logger = logging.getLogger(__name__)


@dataclass
class SentenceConfig:
    """Configuration for sentence generation (latency-sensitive parameters)."""

    tokenizer: str = "nltk"
    language: str = "en"
    minimum_sentence_length: int = 10
    minimum_first_fragment_length: int = 10
    fast_sentence_fragment: bool = True
    context_size: int = 12


_tokenizer_initialized: dict = {}


def init_tokenizer(tokenizer: str = "nltk", language: str = "en") -> None:
    """
    Initialize the stream2sentence tokenizer.

    Call this once at startup to avoid initialization latency on first use.
    """
    key = (tokenizer, language)
    if key not in _tokenizer_initialized:
        s2s.init_tokenizer(tokenizer, language)
        _tokenizer_initialized[key] = True
        logger.info(f"Initialized tokenizer: {tokenizer} for language: {language}")


async def generate_sentences(
    token_stream: AsyncIterator[str],
    sentence_queue: asyncio.Queue,
    config: Optional[SentenceConfig] = None,
    on_sentence: Optional[Callable[[str], None]] = None,
    stop_event: Optional[asyncio.Event] = None,
) -> AsyncIterator[str]:
    """
    Convert an async token stream into sentences.

    Collects tokens from an async LLM response stream, processes them through
    stream2sentence (running in a thread), and yields complete sentences.
    Sentences are also placed into the provided queue for downstream TTS.

    Args:
        token_stream: Async iterator yielding text tokens from LLM
        sentence_queue: asyncio.Queue to receive sentences for TTS processing
        config: Sentence generation configuration (uses defaults if None)
        on_sentence: Optional callback invoked when each sentence is ready
        stop_event: Optional event to signal early termination

    Yields:
        Complete sentences as strings

    Example:
        async def get_llm_tokens():
            async for chunk in openai_stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        sentence_queue = asyncio.Queue()
        async for sentence in generate_sentences(get_llm_tokens(), sentence_queue):
            print(sentence)
    """
    config = config or SentenceConfig()
    stop_event = stop_event or asyncio.Event()

    # Ensure tokenizer is initialized
    init_tokenizer(config.tokenizer, config.language)

    # Thread-safe buffer for bridging async tokens → sync iterator
    char_buffer: queue.Queue = queue.Queue()

    # Internal queue for sentences from the thread
    internal_queue: asyncio.Queue = asyncio.Queue()

    async def collect_tokens():
        """Async task: collect tokens from LLM stream into char buffer."""
        try:
            async for token in token_stream:
                if stop_event.is_set():
                    break
                for char in token:
                    char_buffer.put(char)
        except Exception as e:
            logger.warning(f"Error collecting tokens: {e}")
        finally:
            char_buffer.put(None)  # Sentinel to signal end

    def process_sentences_sync():
        """Sync function: run stream2sentence in thread, emit sentences."""
        loop = asyncio.get_running_loop()

        def char_iterator():
            """Blocking iterator over the character buffer."""
            while True:
                if stop_event.is_set():
                    break
                try:
                    char = char_buffer.get(timeout=0.05)
                    if char is None:
                        break
                    yield char
                except queue.Empty:
                    continue

        try:
            for sentence in s2s.generate_sentences(
                char_iterator(),
                tokenizer=config.tokenizer,
                language=config.language,
                minimum_sentence_length=config.minimum_sentence_length,
                minimum_first_fragment_length=config.minimum_first_fragment_length,
                quick_yield_single_sentence_fragment=config.fast_sentence_fragment,
                context_size=config.context_size,
                cleanup_text_links=True,
                cleanup_text_emojis=True,
            ):
                sentence = sentence.strip()
                if sentence:
                    asyncio.run_coroutine_threadsafe(
                        internal_queue.put(sentence), loop
                    ).result()
        except Exception as e:
            logger.warning(f"Error in sentence generation: {e}")
        finally:
            asyncio.run_coroutine_threadsafe(
                internal_queue.put(None), loop
            ).result()

    # Start concurrent tasks
    loop = asyncio.get_running_loop()
    sentence_task = loop.run_in_executor(None, process_sentences_sync)
    collect_task = asyncio.create_task(collect_tokens())

    try:
        while not stop_event.is_set():
            sentence = await internal_queue.get()
            if sentence is None:
                break

            # Invoke callback if provided
            if on_sentence:
                try:
                    on_sentence(sentence)
                except Exception as e:
                    logger.warning(f"Error in on_sentence callback: {e}")

            # Put on external queue for TTS
            await sentence_queue.put(sentence)

            yield sentence
    finally:
        stop_event.set()
        await collect_task
        await sentence_task


async def collect_and_queue_sentences(
    token_stream: AsyncIterator[str],
    sentence_queue: asyncio.Queue,
    config: Optional[SentenceConfig] = None,
    on_sentence: Optional[Callable[[str], None]] = None,
    stop_event: Optional[asyncio.Event] = None,
) -> str:
    """
    Process a token stream into sentences, queuing them for TTS.

    This is a convenience wrapper around generate_sentences() that consumes
    the entire stream and returns the accumulated text.

    Args:
        token_stream: Async iterator yielding text tokens from LLM
        sentence_queue: asyncio.Queue to receive sentences for TTS processing
        config: Sentence generation configuration
        on_sentence: Optional callback invoked when each sentence is ready
        stop_event: Optional event to signal early termination

    Returns:
        The complete accumulated text from all sentences
    """
    sentences = []
    async for sentence in generate_sentences(
        token_stream,
        sentence_queue,
        config=config,
        on_sentence=on_sentence,
        stop_event=stop_event,
    ):
        sentences.append(sentence)

    return " ".join(sentences)
