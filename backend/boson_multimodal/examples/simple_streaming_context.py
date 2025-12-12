"""
Simple example of streaming TTS with consistent voice using rolling context.

This demonstrates the most straightforward usage of generate_streaming_with_context.
"""

import asyncio
import torch
import soundfile as sf
from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
from boson_multimodal.model.higgs_audio.utils import revert_delay_pattern


async def main():
    # Initialize the serve engine
    device = "cuda" if torch.cuda.is_available() else "cpu"
    serve_engine = HiggsAudioServeEngine(
        model_name_or_path="bosonai/higgs-audio-v2-generation-3B-base",
        audio_tokenizer_name_or_path="bosonai/higgs-audio-v2-tokenizer",
        device=device,
    )

    # Define text chunks to generate audio for
    text_chunks = [
        "Hello, welcome to our podcast.",
        "Today we're discussing AI advancements.",
        "Let's dive into the details.",
    ]

    # Define voice characteristics
    scene_prompt = "Audio is recorded from a quiet room."
    speaker_descriptions = "masculine;moderate pitch;professional;clear"

    # Accumulate audio tokens
    audio_tokens = []

    # Stream generation with rolling context
    async for delta in serve_engine.generate_streaming_with_context(
        text_chunks=text_chunks,
        scene_prompt=scene_prompt,
        speaker_descriptions=speaker_descriptions,
        generation_chunk_buffer_size=5,  # Keep last 5 chunks in context
        temperature=0.7,
        force_audio_gen=True,
    ):
        # Print any generated text
        if delta.text is not None:
            print(delta.text, end="", flush=True)

        # Collect audio tokens
        if delta.audio_tokens is not None:
            audio_tokens.append(delta.audio_tokens)

    print()  # Newline

    # Convert audio tokens to waveform
    if audio_tokens:
        audio_tensor = torch.stack(audio_tokens, dim=1)
        vq_code = revert_delay_pattern(audio_tensor).clip(0, serve_engine.audio_codebook_size - 1)[:, 1:-1]
        waveform = serve_engine.audio_tokenizer.decode(vq_code.unsqueeze(0))[0, 0]

        # Save to file
        output_path = "output_streaming_context.wav"
        sf.write(output_path, waveform, serve_engine.audio_tokenizer.sampling_rate)
        print(f"Saved audio to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
