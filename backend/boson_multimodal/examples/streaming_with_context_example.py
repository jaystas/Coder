"""
Example script demonstrating streaming TTS with rolling context for consistent voice generation.

This example shows how to use the generate_streaming_with_context API to generate
audio across multiple text chunks while maintaining voice consistency through a
rolling context buffer.
"""

import asyncio
import sys
import numpy as np
import soundfile as sf
import torch
from loguru import logger

from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
from boson_multimodal.model.higgs_audio.utils import revert_delay_pattern


MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"


async def example_single_speaker_no_reference():
    """
    Example 1: Single speaker generation with scene prompt but no reference audio.
    The model will choose a consistent voice based on the speaker description.
    """
    logger.info("=" * 80)
    logger.info("Example 1: Single speaker with scene description (no reference audio)")
    logger.info("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    serve_engine = HiggsAudioServeEngine(MODEL_PATH, AUDIO_TOKENIZER_PATH, device=device)

    text_chunks = [
        "Hello, welcome to our podcast about artificial intelligence.",
        "Today we're going to explore the latest developments in machine learning.",
        "Let's start by discussing neural networks and how they work.",
    ]

    scene_prompt = "Audio is recorded from a quiet room with professional microphone."
    speaker_descriptions = "masculine;moderate pitch;clear;professional"

    audio_token_buffer = []

    async for delta in serve_engine.generate_streaming_with_context(
        text_chunks=text_chunks,
        scene_prompt=scene_prompt,
        speaker_descriptions=speaker_descriptions,
        generation_chunk_buffer_size=5,
        temperature=0.7,
        top_p=0.95,
        force_audio_gen=True,
    ):
        if delta.text is not None:
            print(delta.text, end="", flush=True)

        if delta.audio_tokens is not None:
            audio_token_buffer.append(delta.audio_tokens)

    print()  # Newline after text output

    # Process accumulated audio tokens to waveform
    if audio_token_buffer:
        logger.info(f"Processing {len(audio_token_buffer)} audio tokens...")
        audio_tensor = torch.stack(audio_token_buffer, dim=1)
        vq_code = revert_delay_pattern(audio_tensor).clip(0, serve_engine.audio_codebook_size - 1)[:, 1:-1]
        waveform = serve_engine.audio_tokenizer.decode(vq_code.unsqueeze(0))[0, 0]

        output_path = "output_example1_single_speaker.wav"
        sf.write(output_path, waveform, serve_engine.audio_tokenizer.sampling_rate)
        logger.info(f"Saved audio to {output_path}")


async def example_voice_cloning():
    """
    Example 2: Voice cloning with reference audio.
    The model will maintain the voice from the reference audio across all chunks.
    """
    logger.info("=" * 80)
    logger.info("Example 2: Voice cloning with reference audio")
    logger.info("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    serve_engine = HiggsAudioServeEngine(MODEL_PATH, AUDIO_TOKENIZER_PATH, device=device)

    text_chunks = [
        "This is an example of voice cloning.",
        "Notice how the voice remains consistent across these different sentences.",
        "The rolling context buffer helps maintain this consistency.",
    ]

    # Specify reference audio path (replace with actual path)
    # For this example, we'll use one from the voice_prompts directory
    ref_audio_paths = ["boson_multimodal/examples/voice_prompts/belinda.wav"]

    scene_prompt = "Audio is recorded from a quiet indoor environment."

    audio_token_buffer = []

    async for delta in serve_engine.generate_streaming_with_context(
        text_chunks=text_chunks,
        scene_prompt=scene_prompt,
        ref_audio_paths=ref_audio_paths,
        generation_chunk_buffer_size=5,
        temperature=0.3,  # Lower temperature for more consistent voice cloning
        top_p=0.95,
        force_audio_gen=True,
    ):
        if delta.text is not None:
            print(delta.text, end="", flush=True)

        if delta.audio_tokens is not None:
            audio_token_buffer.append(delta.audio_tokens)

    print()

    # Process accumulated audio tokens
    if audio_token_buffer:
        logger.info(f"Processing {len(audio_token_buffer)} audio tokens...")
        audio_tensor = torch.stack(audio_token_buffer, dim=1)
        vq_code = revert_delay_pattern(audio_tensor).clip(0, serve_engine.audio_codebook_size - 1)[:, 1:-1]
        waveform = serve_engine.audio_tokenizer.decode(vq_code.unsqueeze(0))[0, 0]

        output_path = "output_example2_voice_cloning.wav"
        sf.write(output_path, waveform, serve_engine.audio_tokenizer.sampling_rate)
        logger.info(f"Saved audio to {output_path}")


async def example_multi_speaker():
    """
    Example 3: Multi-speaker dialogue with different speaker descriptions.
    Each speaker gets a distinct voice description.
    """
    logger.info("=" * 80)
    logger.info("Example 3: Multi-speaker dialogue")
    logger.info("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    serve_engine = HiggsAudioServeEngine(MODEL_PATH, AUDIO_TOKENIZER_PATH, device=device)

    text_chunks = [
        "[SPEAKER0] Hello! How are you today?",
        "[SPEAKER1] I'm doing great, thanks for asking!",
        "[SPEAKER0] That's wonderful to hear. What have you been working on?",
        "[SPEAKER1] I've been exploring some fascinating research in AI.",
    ]

    scene_prompt = "Two people having a friendly conversation in a quiet room."
    speaker_descriptions = [
        "SPEAKER0: feminine;warm;friendly;moderate pitch",
        "SPEAKER1: masculine;deep voice;calm;professional",
    ]

    audio_token_buffer = []

    async for delta in serve_engine.generate_streaming_with_context(
        text_chunks=text_chunks,
        scene_prompt=scene_prompt,
        speaker_descriptions=speaker_descriptions,
        generation_chunk_buffer_size=5,
        temperature=0.7,
        top_p=0.95,
        force_audio_gen=True,
    ):
        if delta.text is not None:
            print(delta.text, end="", flush=True)

        if delta.audio_tokens is not None:
            audio_token_buffer.append(delta.audio_tokens)

    print()

    # Process accumulated audio tokens
    if audio_token_buffer:
        logger.info(f"Processing {len(audio_token_buffer)} audio tokens...")
        audio_tensor = torch.stack(audio_token_buffer, dim=1)
        vq_code = revert_delay_pattern(audio_tensor).clip(0, serve_engine.audio_codebook_size - 1)[:, 1:-1]
        waveform = serve_engine.audio_tokenizer.decode(vq_code.unsqueeze(0))[0, 0]

        output_path = "output_example3_multi_speaker.wav"
        sf.write(output_path, waveform, serve_engine.audio_tokenizer.sampling_rate)
        logger.info(f"Saved audio to {output_path}")


async def example_streaming_pcm_output():
    """
    Example 4: Streaming PCM output for real-time playback.
    This example demonstrates how to process audio tokens in chunks for lower latency.
    """
    logger.info("=" * 80)
    logger.info("Example 4: Streaming PCM output with chunk processing")
    logger.info("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    serve_engine = HiggsAudioServeEngine(MODEL_PATH, AUDIO_TOKENIZER_PATH, device=device)

    text_chunks = [
        "This example demonstrates streaming audio output.",
        "Audio tokens are processed in chunks for lower latency.",
        "This is useful for real-time applications.",
    ]

    scene_prompt = "Audio is recorded from a quiet room."
    speaker_descriptions = "feminine;clear;moderate pitch"

    audio_token_buffer = []
    chunk_size = 64  # Process audio every 64 tokens
    all_waveforms = []

    async for delta in serve_engine.generate_streaming_with_context(
        text_chunks=text_chunks,
        scene_prompt=scene_prompt,
        speaker_descriptions=speaker_descriptions,
        generation_chunk_buffer_size=5,
        temperature=0.7,
        force_audio_gen=True,
    ):
        if delta.text is not None:
            print(delta.text, end="", flush=True)

        if delta.audio_tokens is not None:
            audio_token_buffer.append(delta.audio_tokens)

            # Process in chunks for streaming playback
            if len(audio_token_buffer) >= chunk_size:
                audio_chunk = torch.stack(audio_token_buffer[:chunk_size], dim=1)
                vq_code = revert_delay_pattern(audio_chunk).clip(0, serve_engine.audio_codebook_size - 1)[:, 1:-1]
                waveform_chunk = serve_engine.audio_tokenizer.decode(vq_code.unsqueeze(0))[0, 0]
                all_waveforms.append(waveform_chunk)

                # Here you could stream to audio output device
                logger.debug(f"Processed chunk: {waveform_chunk.shape} samples")

                # Keep overlap for continuity
                audio_token_buffer = audio_token_buffer[chunk_size:]

    # Process remaining tokens
    if audio_token_buffer:
        audio_chunk = torch.stack(audio_token_buffer, dim=1)
        vq_code = revert_delay_pattern(audio_chunk).clip(0, serve_engine.audio_codebook_size - 1)[:, 1:-1]
        waveform_chunk = serve_engine.audio_tokenizer.decode(vq_code.unsqueeze(0))[0, 0]
        all_waveforms.append(waveform_chunk)

    print()

    # Concatenate all waveforms
    if all_waveforms:
        final_waveform = np.concatenate(all_waveforms)
        output_path = "output_example4_streaming_pcm.wav"
        sf.write(output_path, final_waveform, serve_engine.audio_tokenizer.sampling_rate)
        logger.info(f"Saved audio to {output_path}")


async def main():
    """Run all examples."""
    logger.info("Running streaming TTS with context examples...")

    # Run examples
    await example_single_speaker_no_reference()
    await example_voice_cloning()
    await example_multi_speaker()
    await example_streaming_pcm_output()

    logger.info("All examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
