import asyncio
import base64
import torch
import numpy as np
from io import BytesIO
from dataclasses import dataclass
from typing import List, Optional, Union
from copy import deepcopy
from transformers import AutoTokenizer, AutoProcessor
from transformers.cache_utils import StaticCache
from transformers.generation.streamers import BaseStreamer
from transformers.generation.stopping_criteria import StoppingCriteria
from dataclasses import asdict
from loguru import logger
import threading
import librosa


from ..dataset.chatml_dataset import ChatMLSample, ChatMLDatasetSample, prepare_chatml_sample
from ..model.higgs_audio import HiggsAudioModel
from ..model.higgs_audio.utils import revert_delay_pattern
from ..data_collator.higgs_audio_collator import HiggsAudioSampleCollator
from ..audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
from ..data_types import Message, AudioContent, TextContent


@dataclass
class HiggsAudioStreamerDelta:
    """Represents a chunk of generated content, either text or audio tokens."""

    text: Optional[str] = None
    text_tokens: Optional[torch.Tensor] = None
    audio_tokens: Optional[torch.Tensor] = None
    finish_reason: Optional[str] = None


class AsyncHiggsAudioStreamer(BaseStreamer):
    """
    Async streamer that handles both text and audio token generation from Higgs-Audio model.
    Stores chunks in a queue to be consumed by downstream applications.

    Parameters:
        tokenizer (`AutoTokenizer`):
            The tokenizer used to decode text tokens.
        skip_prompt (`bool`, *optional*, defaults to `False`):
            Whether to skip the prompt tokens in generation.
        timeout (`float`, *optional*):
            The timeout for the queue. If `None`, the queue will block indefinitely.
        decode_kwargs (`dict`, *optional*):
            Additional keyword arguments to pass to the tokenizer's `decode` method.

    Examples:
        ```python
        >>> from transformers import AutoTokenizer
        >>> from threading import Thread
        >>> import asyncio

        >>> tokenizer = AutoTokenizer.from_pretrained("path/to/higgs/tokenizer")
        >>> model = HiggsAudioModel.from_pretrained("path/to/higgs/model")
        >>> inputs = tokenizer(["Generate some text and audio:"], return_tensors="pt")

        >>> async def main():
        ...     streamer = AsyncHiggsAudioStreamer(tokenizer)
        ...     generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=20)
        ...     thread = Thread(target=model.generate, kwargs=generation_kwargs)
        ...     thread.start()
        ...
        ...     async for delta in streamer:
        ...         if delta.text is not None:
        ...             print("Text:", delta.text)
        ...         if delta.audio_tokens is not None:
        ...             print("Audio tokens shape:", delta.audio_tokens.shape)
        >>> asyncio.run(main())
        ```
    """

    def __init__(
        self,
        tokenizer: "AutoTokenizer",
        skip_prompt: bool = False,
        timeout: Optional[float] = None,
        audio_num_codebooks: int = 1,
        **decode_kwargs,
    ):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.timeout = timeout
        self.decode_kwargs = decode_kwargs
        self.audio_num_codebooks = audio_num_codebooks
        # Queue to store generated chunks
        self.queue = asyncio.Queue()
        self.stop_signal = None

        # Get running event loop
        self.loop = asyncio.get_running_loop()
        self.has_asyncio_timeout = hasattr(asyncio, "timeout")

        # State tracking
        self.next_tokens_are_prompt = True

    def put(self, value: torch.Tensor):
        """
        Receives tokens and processes them as either text or audio tokens.
        For text tokens, decodes and caches them until complete words are formed.
        For audio tokens, directly queues them.
        """
        if value.shape[0] > 1 and not self.next_tokens_are_prompt:
            # This is likely audio tokens (shape: [audio_num_codebooks])
            assert value.shape[0] == self.audio_num_codebooks, "Number of codebooks mismatch"
            delta = HiggsAudioStreamerDelta(audio_tokens=value)
            if self.loop.is_running():
                self.loop.call_soon_threadsafe(self.queue.put_nowait, delta)
            return

        # Skip prompt tokens if configured
        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        # Process as text tokens
        if len(value.shape) > 1:
            value = value[0]

        text = self.tokenizer.decode(value, **self.decode_kwargs)
        delta = HiggsAudioStreamerDelta(text=text, text_tokens=value)
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.queue.put_nowait, delta)

    def end(self):
        """Flushes any remaining text tokens and signals the end of generation."""
        self.next_tokens_are_prompt = True
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.queue.put_nowait, self.stop_signal)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            if self.has_asyncio_timeout:
                async with asyncio.timeout(self.timeout):
                    value = await self.queue.get()
            else:
                value = await asyncio.wait_for(self.queue.get(), timeout=self.timeout)
        except asyncio.TimeoutError:
            raise TimeoutError()
        else:
            if value == self.stop_signal:
                raise StopAsyncIteration()
            else:
                return value


class AsyncStoppingCriteria(StoppingCriteria):
    """
    Stopping criteria that checks for stop signal from a threading event.

    Args:
        stop_signal (threading.Event): Event that will receive stop signals
    """

    def __init__(self, stop_signal: threading.Event):
        self.stop_signal = stop_signal

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        if self.stop_signal.is_set():
            logger.info(f"Stop signal received. Can be caused by client disconnection.")
            return True
        return False


@dataclass
class HiggsAudioResponse:
    audio: Optional[np.ndarray] = None
    generated_audio_tokens: Optional[np.ndarray] = None
    sampling_rate: Optional[int] = None
    generated_text: str = ""
    generated_text_tokens: Optional[np.ndarray] = None
    usage: Optional[dict] = None


class StreamingVoiceGenerator:
    """
    Manages rolling context buffer for consistent voice generation across multiple text chunks.

    This class maintains:
    - Base messages (system prompt with scene description and speaker descriptions)
    - Reference audio token IDs (for voice cloning)
    - Rolling buffer of generated audio IDs from previous chunks
    - Rolling buffer of generation messages (user/assistant pairs)

    The rolling context ensures voice consistency by accumulating audio_ids and input tokens
    from previous generations as context for subsequent chunks.
    """

    def __init__(
        self,
        serve_engine: "HiggsAudioServeEngine",
        scene_prompt: Optional[str] = None,
        speaker_descriptions: Optional[Union[str, List[str]]] = None,
        ref_audio_paths: Optional[List[str]] = None,
        generation_chunk_buffer_size: Optional[int] = 5,
    ):
        """
        Initialize the streaming voice generator with context management.

        Args:
            serve_engine: The HiggsAudioServeEngine instance to use for generation
            scene_prompt: Scene description to include in system message (e.g., "Audio is recorded from a quiet room")
            speaker_descriptions: Speaker voice descriptions. Can be:
                - Single string for single speaker (e.g., "masculine;moderate pitch;professional")
                - List of strings for multiple speakers (e.g., ["SPEAKER0: feminine;warm", "SPEAKER1: masculine;deep"])
            ref_audio_paths: Optional list of reference audio file paths for voice cloning
            generation_chunk_buffer_size: Maximum number of generated chunks to keep in rolling buffer.
                Set to None for unlimited buffer (not recommended for long generations)
        """
        self.serve_engine = serve_engine
        self.scene_prompt = scene_prompt
        self.speaker_descriptions = speaker_descriptions
        self.generation_chunk_buffer_size = generation_chunk_buffer_size

        # Initialize buffers
        self.base_messages = []  # System message + reference audio messages (if any)
        self.ref_audio_ids = []  # Reference audio token IDs (immutable)
        self.generated_audio_ids = []  # Rolling buffer of generated audio IDs
        self.generation_messages = []  # Rolling buffer of generation messages

        # Prepare base context
        self._prepare_base_context(ref_audio_paths)

    def _prepare_base_context(self, ref_audio_paths: Optional[List[str]]):
        """
        Prepare the base context including system message and reference audio.

        Args:
            ref_audio_paths: List of paths to reference audio files for voice cloning
        """
        # Build system message with scene description and/or speaker descriptions
        system_content_parts = []

        if self.scene_prompt or self.speaker_descriptions:
            system_content_parts.append("Generate audio following instruction.\n\n<|scene_desc_start|>")

            if self.scene_prompt:
                system_content_parts.append(self.scene_prompt)

            # Add speaker descriptions
            if self.speaker_descriptions:
                if isinstance(self.speaker_descriptions, str):
                    # Single speaker description
                    system_content_parts.append(f"\n{self.speaker_descriptions}")
                elif isinstance(self.speaker_descriptions, list):
                    # Multiple speaker descriptions
                    system_content_parts.append("\n" + "\n".join(self.speaker_descriptions))

            system_content_parts.append("\n<|scene_desc_end|>")

            system_message = Message(
                role="system",
                content="".join(system_content_parts),
            )
            self.base_messages.append(system_message)

        # Load and encode reference audio files if provided
        if ref_audio_paths:
            for audio_path in ref_audio_paths:
                # Load the reference audio
                raw_audio, _ = librosa.load(audio_path, sr=self.serve_engine.audio_tokenizer.sampling_rate)

                # Encode to audio tokens
                audio_ids = self.serve_engine.audio_tokenizer.encode(
                    raw_audio, self.serve_engine.audio_tokenizer.sampling_rate
                )
                self.ref_audio_ids.append(audio_ids.squeeze(0).cpu())

                # Add reference audio to messages as user/assistant pair
                # This provides the model with voice examples
                self.base_messages.extend(
                    [
                        Message(role="user", content="Generate audio in this voice style."),
                        Message(role="assistant", content=AudioContent(audio_url=audio_path)),
                    ]
                )

    def _build_chatml_sample(
        self, messages: List[Message], context_audio_ids: List[torch.Tensor]
    ) -> ChatMLDatasetSample:
        """
        Build a ChatMLDatasetSample from messages and context audio IDs.

        Args:
            messages: List of Message objects representing the conversation
            context_audio_ids: List of audio token tensors (reference + generated from previous chunks)

        Returns:
            ChatMLDatasetSample ready for generation
        """
        # Prepare the ChatML sample
        chatml_sample = ChatMLSample(messages=messages)

        # Convert to tokens using prepare_chatml_sample
        input_tokens, _, _, _ = prepare_chatml_sample(chatml_sample, self.serve_engine.tokenizer)

        # Build audio_ids_concat and audio_ids_start from context_audio_ids
        if context_audio_ids:
            audio_ids_concat = torch.cat([audio_id.cpu() for audio_id in context_audio_ids], dim=1)
            audio_ids_start = torch.cumsum(
                torch.tensor([0] + [audio_id.shape[1] for audio_id in context_audio_ids], dtype=torch.long), dim=0
            )[:-1]
        else:
            audio_ids_concat = None
            audio_ids_start = None

        # Create the dataset sample
        sample = ChatMLDatasetSample(
            input_ids=torch.LongTensor(input_tokens),
            label_ids=None,
            audio_ids_concat=audio_ids_concat,
            audio_ids_start=audio_ids_start,
            audio_waveforms_concat=None,
            audio_waveforms_start=None,
            audio_sample_rate=None,
            audio_speaker_indices=None,
        )

        return sample

    def _process_audio_tokens(self, chunk_audio_tokens: List[torch.Tensor]) -> torch.Tensor:
        """
        Process accumulated audio tokens from a chunk (stack, revert delay pattern, clip).

        Args:
            chunk_audio_tokens: List of audio token tensors from streaming generation

        Returns:
            Processed audio token tensor ready to be added to context buffer
        """
        if not chunk_audio_tokens:
            return None

        # Stack tokens along time dimension: (num_codebooks, seq_len)
        audio_tensor = torch.stack(chunk_audio_tokens, dim=1)

        # Revert delay pattern to get original token format
        audio_out_ids = revert_delay_pattern(audio_tensor).clip(
            0, self.serve_engine.audio_codebook_size - 1
        )[:, 1:-1]

        return audio_out_ids

    def _trim_buffers(self):
        """Trim the rolling buffers to maximum size to prevent unbounded memory growth."""
        if self.generation_chunk_buffer_size is not None and len(self.generated_audio_ids) > self.generation_chunk_buffer_size:
            # Keep only the last N chunks
            self.generated_audio_ids = self.generated_audio_ids[-self.generation_chunk_buffer_size :]

            # Keep last N*2 messages (each chunk adds user + assistant message)
            self.generation_messages = self.generation_messages[(-2 * self.generation_chunk_buffer_size) :]

    async def generate_streaming(
        self,
        text_chunks: List[str],
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_k: Optional[int] = None,
        top_p: float = 0.95,
        stop_strings: Optional[List[str]] = None,
        force_audio_gen: bool = True,
        ras_win_len: Optional[int] = 7,
        ras_win_max_num_repeat: int = 2,
        seed: Optional[int] = None,
    ):
        """
        Generate streaming audio across multiple text chunks with rolling context for voice consistency.

        This method:
        1. For each text chunk:
           a. Builds messages including base context + rolling generation history
           b. Includes accumulated audio_ids from previous chunks as context
           c. Streams audio tokens via generate_delta_stream
           d. Updates rolling buffers with generated audio and messages
           e. Trims buffers if they exceed maximum size
        2. Yields HiggsAudioStreamerDelta objects for each generated token

        Args:
            text_chunks: List of text strings to generate audio for
            max_new_tokens: Maximum new tokens per chunk
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            stop_strings: List of strings that stop generation
            force_audio_gen: Whether to force audio generation (vs text)
            ras_win_len: RAS (Repetition Aware Sampling) window length
            ras_win_max_num_repeat: Maximum repetitions in RAS window
            seed: Random seed for generation

        Yields:
            HiggsAudioStreamerDelta: Streaming deltas containing text and/or audio tokens
        """
        if stop_strings is None:
            stop_strings = ["<|end_of_text|>", "<|eot_id|>"]

        for chunk_idx, chunk_text in enumerate(text_chunks):
            logger.info(f"Processing chunk {chunk_idx + 1}/{len(text_chunks)}: {chunk_text[:50]}...")

            # Build messages for this chunk: base + previous generations + new chunk
            chunk_message = Message(role="user", content=chunk_text)
            messages = self.base_messages + self.generation_messages + [chunk_message]

            # Build context audio: reference audio + generated audio from previous chunks
            context_audio_ids = self.ref_audio_ids + self.generated_audio_ids

            # Build ChatMLDatasetSample
            sample = self._build_chatml_sample(messages, context_audio_ids)

            # Collate the sample
            batch_data = self.serve_engine.collator([sample])
            batch = asdict(batch_data)
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.contiguous().to(self.serve_engine.device)

            # Prepare inputs for generation
            postfix = "<|start_header_id|>assistant<|end_header_id|>\n\n"
            if force_audio_gen:
                postfix += "<|audio_out_bos|>"
            postfix_tokens = self.serve_engine.tokenizer.encode(postfix, add_special_tokens=False)

            # Add postfix to input_ids
            batch["input_ids"] = torch.cat(
                [batch["input_ids"], torch.tensor([postfix_tokens], device=self.serve_engine.device)], dim=1
            )

            # Reset KV caches for this chunk
            self.serve_engine._prepare_kv_caches()

            # Stream this chunk
            chunk_audio_tokens = []

            with torch.no_grad():
                streamer = AsyncHiggsAudioStreamer(
                    self.serve_engine.tokenizer,
                    audio_num_codebooks=self.serve_engine.model.config.audio_num_codebooks,
                    skip_prompt=True,
                )
                generation_kwargs = dict(
                    **batch,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    stop_strings=stop_strings,
                    tokenizer=self.serve_engine.tokenizer,
                    do_sample=False if temperature == 0.0 else True,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    past_key_values_buckets=self.serve_engine.kv_caches,
                    ras_win_len=ras_win_len,
                    ras_win_max_num_repeat=ras_win_max_num_repeat,
                    seed=seed,
                    streamer=streamer,
                )
                thread = threading.Thread(target=self.serve_engine.model.generate, kwargs=generation_kwargs)
                thread.start()

                async for delta in streamer:
                    # Accumulate audio tokens locally for context buffer
                    if delta.audio_tokens is not None:
                        chunk_audio_tokens.append(delta.audio_tokens.cpu())

                    # Yield to client for streaming playback
                    yield delta

            # After chunk completes, update buffers
            if chunk_audio_tokens:
                # Process accumulated tokens
                audio_out_ids = self._process_audio_tokens(chunk_audio_tokens)

                if audio_out_ids is not None:
                    # Add to rolling buffer
                    self.generated_audio_ids.append(audio_out_ids)
                    self.generation_messages.extend(
                        [
                            chunk_message,
                            Message(role="assistant", content=AudioContent(audio_url="")),
                        ]
                    )

                    # Trim buffers to prevent unbounded growth
                    self._trim_buffers()

                    logger.info(
                        f"Chunk {chunk_idx + 1} complete. Buffer size: {len(self.generated_audio_ids)} chunks, "
                        f"{len(self.generation_messages)} messages"
                    )


class HiggsAudioServeEngine:
    def __init__(
        self,
        model_name_or_path: str,
        audio_tokenizer_name_or_path: str,
        tokenizer_name_or_path: Optional[str] = None,
        device: str = "cuda",
        torch_dtype: Union[torch.dtype, str] = "auto",
        kv_cache_lengths: List[int] = [1024, 4096, 8192],  # Multiple KV cache sizes
    ):
        """
        Initialize the HiggsAudioServeEngine, a serving wrapper for the HiggsAudioModel.
        The model, tokenizer, and audio tokenizer will be downloaded from the Hugging Face Hub if they are not local.

        Args:
            model_name_or_path (str):
                The name or path of the model to load.
            audio_tokenizer_name_or_path (str):
                The name or path of the audio tokenizer to load.
            tokenizer_name_or_path (str):
                The name or path of the tokenizer to load.
            device (str):
                The device to use for the model.
            kv_cache_lengths (List[int]):
                The lengths of the KV caches to use for the model. Used for cuda graph capture when device is cuda.
            torch_dtype (Union[torch.dtype, str]):
                The dtype to use for the model.
        """
        self.device = device
        self.model_name_or_path = model_name_or_path
        self.torch_dtype = torch_dtype

        # Initialize model and tokenizer
        self.model = HiggsAudioModel.from_pretrained(model_name_or_path, torch_dtype=torch_dtype).to(device)
        logger.info(f"Loaded model from {model_name_or_path}, dtype: {self.model.dtype}")

        if tokenizer_name_or_path is None:
            tokenizer_name_or_path = model_name_or_path
        logger.info(f"Loading tokenizer from {tokenizer_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

        logger.info(f"Initializing Higgs Audio Tokenizer")
        self.audio_tokenizer = load_higgs_audio_tokenizer(audio_tokenizer_name_or_path, device=device)

        self.audio_num_codebooks = self.model.config.audio_num_codebooks
        self.audio_codebook_size = self.model.config.audio_codebook_size
        self.audio_tokenizer_tps = self.audio_tokenizer.tps
        self.samples_per_token = int(self.audio_tokenizer.sampling_rate // self.audio_tokenizer_tps)
        self.hamming_window_len = 2 * self.audio_num_codebooks * self.samples_per_token
        # Set the audio special tokens
        self.model.set_audio_special_tokens(self.tokenizer)

        # Prepare KV caches for different lengths
        cache_config = deepcopy(self.model.config.text_config)
        cache_config.num_hidden_layers = self.model.config.text_config.num_hidden_layers
        if self.model.config.audio_dual_ffn_layers:
            cache_config.num_hidden_layers += len(self.model.config.audio_dual_ffn_layers)
        # A list of KV caches for different lengths
        self.kv_caches = {
            length: StaticCache(
                config=cache_config,
                max_batch_size=1,
                max_cache_len=length,
                device=self.model.device,
                dtype=self.model.dtype,
            )
            for length in sorted(kv_cache_lengths)
        }

        if self.model.config.encode_whisper_embed:
            logger.info(f"Loading whisper processor")
            whisper_processor = AutoProcessor.from_pretrained(
                "openai/whisper-large-v3-turbo",
                trust_remote=True,
                device=self.device,
            )
        else:
            whisper_processor = None

        # Reuse collator to prepare inference samples
        self.collator = HiggsAudioSampleCollator(
            whisper_processor=whisper_processor,
            encode_whisper_embed=self.model.config.encode_whisper_embed,
            audio_in_token_id=self.model.config.audio_in_token_idx,
            audio_out_token_id=self.model.config.audio_out_token_idx,
            audio_stream_bos_id=self.model.config.audio_stream_bos_id,
            audio_stream_eos_id=self.model.config.audio_stream_eos_id,
            pad_token_id=self.model.config.pad_token_id,
            return_audio_in_tokens=False,
            use_delay_pattern=self.model.config.use_delay_pattern,
            audio_num_codebooks=self.model.config.audio_num_codebooks,
            round_to=1,
        )

        # Capture CUDA graphs for each KV cache length
        if device == "cuda":
            logger.info(f"Capturing CUDA graphs for each KV cache length")
            self.model.capture_model(self.kv_caches.values())

    def _prepare_inputs(self, chat_ml_sample: ChatMLSample, force_audio_gen: bool = False):
        input_tokens, _, audio_contents, _ = prepare_chatml_sample(
            chat_ml_sample,
            self.tokenizer,
        )

        postfix = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        if force_audio_gen:
            postfix += "<|audio_out_bos|>"
        postfix = self.tokenizer.encode(postfix, add_special_tokens=False)
        input_tokens.extend(postfix)

        # Configure the audio inputs
        audio_ids_l = []
        for audio_content in audio_contents:
            if audio_content.audio_url not in ["placeholder", ""]:
                raw_audio, _ = librosa.load(audio_content.audio_url, sr=self.audio_tokenizer.sampling_rate)
            elif audio_content.raw_audio is not None:
                raw_audio, _ = librosa.load(
                    BytesIO(base64.b64decode(audio_content.raw_audio)), sr=self.audio_tokenizer.sampling_rate
                )
            else:
                raw_audio = None

            if raw_audio is not None:
                audio_ids = self.audio_tokenizer.encode(raw_audio, self.audio_tokenizer.sampling_rate)
                audio_ids_l.append(audio_ids.squeeze(0).cpu())

        if len(audio_ids_l) > 0:
            audio_ids_start = torch.tensor(
                np.cumsum(np.array([0] + [audio_ids.shape[1] for audio_ids in audio_ids_l])),
                dtype=torch.long,
                device=self.device,
            )[0:-1]
            audio_ids_concat = torch.cat(audio_ids_l, dim=1)
        else:
            audio_ids_start = None
            audio_ids_concat = None

        sample = ChatMLDatasetSample(
            input_ids=torch.LongTensor(input_tokens),
            label_ids=None,
            audio_ids_concat=audio_ids_concat,
            audio_ids_start=audio_ids_start,
            audio_waveforms_concat=None,
            audio_waveforms_start=None,
            audio_sample_rate=None,
            audio_speaker_indices=None,
        )
        data = self.collator([sample])
        inputs = asdict(data)
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.model.device)

        return inputs

    def _prepare_kv_caches(self):
        for kv_cache in self.kv_caches.values():
            kv_cache.reset()

    def generate(
        self,
        chat_ml_sample: ChatMLSample,
        max_new_tokens: int,
        temperature: float = 0.7,
        top_k: Optional[int] = None,
        top_p: float = 0.95,
        stop_strings: Optional[List[str]] = None,
        force_audio_gen: bool = False,
        ras_win_len: Optional[int] = 7,
        ras_win_max_num_repeat: int = 2,
        seed: Optional[int] = None,
    ):
        """
        Generate audio from a chatml sample.
        Args:
            chat_ml_sample: A chatml sample.
            max_new_tokens: The maximum number of new tokens to generate.
            temperature: The temperature to use for the generation.
            top_p: The top p to use for the generation.
            stop_strings: A list of strings to stop the generation.
            force_audio_gen: Whether to force audio generation. This ensures the model generates audio tokens rather than text tokens.
            ras_win_len: The length of the RAS window. We use 7 by default. You can disable it by setting it to None or <=0.
            ras_win_max_num_repeat: The maximum number of times to repeat the RAS window.
        Returns:
            A dictionary with the following keys:
                audio: The generated audio.
                sampling_rate: The sampling rate of the generated audio.
        """
        # Default stop strings
        if stop_strings is None:
            stop_strings = ["<|end_of_text|>", "<|eot_id|>"]
        if ras_win_len is not None and ras_win_len <= 0:
            ras_win_len = None

        with torch.no_grad():
            inputs = self._prepare_inputs(chat_ml_sample, force_audio_gen=force_audio_gen)
            prompt_token_ids = inputs["input_ids"][0].cpu().numpy()

            self._prepare_kv_caches()

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                stop_strings=stop_strings,
                tokenizer=self.tokenizer,
                do_sample=False if temperature == 0.0 else True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                past_key_values_buckets=self.kv_caches,
                ras_win_len=ras_win_len,
                ras_win_max_num_repeat=ras_win_max_num_repeat,
                seed=seed,
            )

            if len(outputs[1]) > 0:
                wv_list = []
                for output_audio in outputs[1]:
                    vq_code = revert_delay_pattern(output_audio).clip(0, self.audio_codebook_size - 1)[:, 1:-1]
                    wv_numpy = self.audio_tokenizer.decode(vq_code.unsqueeze(0))[0, 0]
                    wv_list.append(wv_numpy)
                wv_numpy = np.concatenate(wv_list)
            else:
                wv_numpy = None

            # We only support one request at a time now
            generated_text_tokens = outputs[0][0].cpu().numpy()[len(prompt_token_ids) :]
            generated_text = self.tokenizer.decode(generated_text_tokens)
            generated_audio_tokens = outputs[1][0].cpu().numpy()
            return HiggsAudioResponse(
                audio=wv_numpy,
                generated_audio_tokens=generated_audio_tokens,
                sampling_rate=self.audio_tokenizer.sampling_rate,
                generated_text=generated_text,
                generated_text_tokens=generated_text_tokens,
                usage={
                    "prompt_tokens": prompt_token_ids.shape[0],
                    "completion_tokens": generated_text_tokens.shape[0] + generated_audio_tokens.shape[1],
                    "total_tokens": (
                        prompt_token_ids.shape[0] + generated_text_tokens.shape[0] + generated_audio_tokens.shape[1]
                    ),
                    "cached_tokens": 0,
                },
            )

    async def generate_delta_stream(
        self,
        chat_ml_sample: ChatMLSample,
        max_new_tokens: int,
        temperature: float = 0.7,
        top_k: Optional[int] = None,
        top_p: float = 0.95,
        stop_strings: Optional[List[str]] = None,
        force_audio_gen: bool = False,
        ras_win_len: Optional[int] = 7,
        ras_win_max_num_repeat: int = 2,
        seed: Optional[int] = None,
    ):
        """
        Generate audio from a chatml sample.
        Args:
            chat_ml_sample: A chatml sample.
            max_new_tokens: The maximum number of new tokens to generate.
            temperature: The temperature to use for the generation.
            top_p: The top p to use for the generation.
            stop_strings: A list of strings to stop the generation.
            force_audio_gen: Whether to force audio generation. This ensures the model generates audio tokens rather than text tokens.
            ras_win_len: The length of the RAS window. We use 7 by default. You can disable it by setting it to None or <=0.
            ras_win_max_num_repeat: The maximum number of times to repeat the RAS window.
        Returns:
             Delta AsyncGenerator
        """
        # Default stop strings
        if stop_strings is None:
            stop_strings = ["<|end_of_text|>", "<|eot_id|>"]
        if ras_win_len is not None and ras_win_len <= 0:
            ras_win_len = None

        with torch.no_grad():
            inputs = self._prepare_inputs(chat_ml_sample, force_audio_gen=force_audio_gen)

            self._prepare_kv_caches()

            streamer = AsyncHiggsAudioStreamer(
                self.tokenizer,
                audio_num_codebooks=self.model.config.audio_num_codebooks,
                skip_prompt=True,
            )
            generation_kwargs = dict(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                stop_strings=stop_strings,
                tokenizer=self.tokenizer,
                do_sample=False if temperature == 0.0 else True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                past_key_values_buckets=self.kv_caches,
                ras_win_len=ras_win_len,
                ras_win_max_num_repeat=ras_win_max_num_repeat,
                seed=seed,
                streamer=streamer,
            )
            thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            async for delta in streamer:
                yield delta

    async def generate_streaming_with_context(
        self,
        text_chunks: List[str],
        scene_prompt: Optional[str] = None,
        speaker_descriptions: Optional[Union[str, List[str]]] = None,
        ref_audio_paths: Optional[List[str]] = None,
        generation_chunk_buffer_size: Optional[int] = 5,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_k: Optional[int] = None,
        top_p: float = 0.95,
        stop_strings: Optional[List[str]] = None,
        force_audio_gen: bool = True,
        ras_win_len: Optional[int] = 7,
        ras_win_max_num_repeat: int = 2,
        seed: Optional[int] = None,
    ):
        """
        Generate streaming audio with rolling context buffer for consistent voice across multiple text chunks.

        This is the high-level API for generating audio with voice consistency. It:
        1. Prepares base context (scene description, speaker descriptions, reference audio)
        2. For each text chunk, generates audio while maintaining rolling context
        3. Accumulates generated audio_ids and messages from previous chunks
        4. Ensures voice consistency by providing this accumulated context to subsequent generations

        Example usage:
            ```python
            serve_engine = HiggsAudioServeEngine(...)

            text_chunks = [
                "Hello, welcome to our podcast.",
                "Today we're discussing AI advancements.",
                "Let's dive into the details."
            ]

            async for delta in serve_engine.generate_streaming_with_context(
                text_chunks=text_chunks,
                scene_prompt="Audio is recorded from a quiet room.",
                speaker_descriptions="masculine;moderate pitch;professional",
                generation_chunk_buffer_size=5,
                temperature=0.7,
            ):
                if delta.audio_tokens is not None:
                    # Process audio tokens for playback
                    process_audio(delta.audio_tokens)
            ```

        Args:
            text_chunks: List of text strings to generate audio for. Can be sentences, paragraphs, or dialogue turns.
            scene_prompt: Optional scene description (e.g., "Audio is recorded from a quiet room").
                Will be included in the system message within <|scene_desc_start|>...<|scene_desc_end|> tags.
            speaker_descriptions: Optional speaker voice descriptions. Can be:
                - Single string for single speaker: "masculine;moderate pitch;professional"
                - List of strings for multiple speakers: ["SPEAKER0: feminine;warm", "SPEAKER1: masculine;deep"]
            ref_audio_paths: Optional list of paths to reference audio files for voice cloning.
                These audio files will be encoded and included in the context to guide voice generation.
            generation_chunk_buffer_size: Maximum number of generated chunks to keep in rolling buffer.
                Larger values provide more context but use more memory. Default is 5.
                Set to None for unlimited buffer (not recommended for very long generations).
            max_new_tokens: Maximum number of new tokens to generate per chunk.
            temperature: Sampling temperature (0.0 = deterministic, higher = more random).
            top_k: Top-k sampling parameter. If None, no top-k filtering is applied.
            top_p: Top-p (nucleus) sampling parameter.
            stop_strings: List of strings that stop generation. Defaults to ["<|end_of_text|>", "<|eot_id|>"].
            force_audio_gen: Whether to force audio generation (vs text). Usually True for TTS.
            ras_win_len: RAS (Repetition Aware Sampling) window length. Default is 7.
                Set to None or <=0 to disable RAS.
            ras_win_max_num_repeat: Maximum number of repetitions allowed in RAS window.
            seed: Optional random seed for reproducible generation.

        Yields:
            HiggsAudioStreamerDelta: Streaming deltas containing text and/or audio tokens.
                Each delta may contain:
                - delta.text: Generated text (if any)
                - delta.audio_tokens: Generated audio tokens (shape: [num_codebooks])
                - delta.finish_reason: Reason for stopping (if generation finished)

        Note:
            This method maintains a rolling context buffer that accumulates:
            - generated_audio_ids: Audio token IDs from previous chunks
            - generation_messages: User/assistant message pairs from previous chunks

            This rolling context is what enables voice consistency across chunks. The buffer is
            automatically trimmed to `generation_chunk_buffer_size` to prevent unbounded growth.
        """
        generator = StreamingVoiceGenerator(
            serve_engine=self,
            scene_prompt=scene_prompt,
            speaker_descriptions=speaker_descriptions,
            ref_audio_paths=ref_audio_paths,
            generation_chunk_buffer_size=generation_chunk_buffer_size,
        )

        async for delta in generator.generate_streaming(
            text_chunks=text_chunks,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_strings=stop_strings,
            force_audio_gen=force_audio_gen,
            ras_win_len=ras_win_len,
            ras_win_max_num_repeat=ras_win_max_num_repeat,
            seed=seed,
        ):
            yield delta
