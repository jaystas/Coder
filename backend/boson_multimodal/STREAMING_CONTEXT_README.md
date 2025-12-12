# Streaming TTS with Rolling Context for Consistent Voice Generation

This feature enables streaming text-to-speech generation with consistent voice characteristics across multiple text chunks using a rolling context buffer.

## Overview

The `generate_streaming_with_context` API provides:

- **Voice Consistency**: Maintains consistent voice across multiple text chunks
- **Rolling Context Buffer**: Accumulates generated audio IDs and messages from previous chunks
- **Streaming Output**: Low-latency streaming via async generators
- **Flexible Configuration**: Supports scene descriptions, speaker descriptions, and reference audio

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Streaming TTS with Rolling Context Buffer                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Initialize context:                                     │
│     - System message (scene + speaker descriptions)         │
│     - Reference audio_ids (if voice cloning)                │
│     - Empty generated_audio_ids buffer                      │
│     - Empty generation_messages buffer                      │
│                                                              │
│  2. For each text chunk:                                    │
│     ┌───────────────────────────────────────────┐          │
│     │ a. Build messages:                         │          │
│     │    messages = base_messages +              │          │
│     │              generation_messages +         │          │
│     │              new_chunk_message             │          │
│     │                                            │          │
│     │ b. Build context audio:                   │          │
│     │    context_audio_ids =                    │          │
│     │        ref_audio_ids + generated_audio_ids│          │
│     │                                            │          │
│     │ c. Call generate_delta_stream()           │          │
│     │                                            │          │
│     │ d. Stream deltas:                         │          │
│     │    - Accumulate audio tokens locally      │          │
│     │    - Yield audio chunks to client         │          │
│     │                                            │          │
│     │ e. After completion:                      │          │
│     │    - Append audio_ids to buffer           │          │
│     │    - Append messages to buffer            │          │
│     │    - Trim buffers if > buffer_size        │          │
│     └───────────────────────────────────────────┘          │
│                                                              │
│  3. Return complete audio stream                            │
└─────────────────────────────────────────────────────────────┘
```

## How It Works

### Rolling Context Buffer

The system maintains two key buffers:

1. **`generated_audio_ids`**: List of audio token tensors from previously generated chunks
2. **`generation_messages`**: List of user/assistant message pairs from previous generations

For each new text chunk:
- The system includes context from previous chunks (audio IDs + messages)
- This accumulated context guides the model to maintain voice consistency
- Buffers are trimmed to `generation_chunk_buffer_size` to prevent unbounded growth

### Voice Consistency Mechanism

Voice consistency is achieved through:

1. **Initial Context**: Scene prompt and speaker descriptions set the base voice characteristics
2. **Audio Context**: Previous generated audio tokens provide voice examples
3. **Message History**: Previous user/assistant pairs maintain conversational context
4. **Buffer Management**: Rolling window keeps recent context relevant

## API Reference

### `HiggsAudioServeEngine.generate_streaming_with_context()`

```python
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
) -> AsyncGenerator[HiggsAudioStreamerDelta, None]:
```

#### Parameters

- **`text_chunks`** (List[str]): List of text strings to generate audio for
- **`scene_prompt`** (Optional[str]): Scene description (e.g., "Audio is recorded from a quiet room")
- **`speaker_descriptions`** (Optional[Union[str, List[str]]]):
  - Single string: `"masculine;moderate pitch;professional"`
  - List for multi-speaker: `["SPEAKER0: feminine;warm", "SPEAKER1: masculine;deep"]`
- **`ref_audio_paths`** (Optional[List[str]]): Paths to reference audio files for voice cloning
- **`generation_chunk_buffer_size`** (Optional[int]): Maximum chunks to keep in rolling buffer (default: 5)
- **`max_new_tokens`** (int): Maximum tokens per chunk (default: 2048)
- **`temperature`** (float): Sampling temperature (default: 0.7)
- **`top_k`** (Optional[int]): Top-k sampling parameter
- **`top_p`** (float): Top-p (nucleus) sampling (default: 0.95)
- **`stop_strings`** (Optional[List[str]]): Strings that stop generation
- **`force_audio_gen`** (bool): Force audio generation vs text (default: True)
- **`ras_win_len`** (Optional[int]): RAS window length (default: 7)
- **`ras_win_max_num_repeat`** (int): Max RAS repetitions (default: 2)
- **`seed`** (Optional[int]): Random seed for reproducibility

#### Returns

AsyncGenerator yielding `HiggsAudioStreamerDelta` objects with:
- `delta.text`: Generated text (if any)
- `delta.audio_tokens`: Audio tokens (shape: [num_codebooks])
- `delta.finish_reason`: Reason for stopping (if finished)

## Usage Examples

### Example 1: Basic Usage with Scene and Speaker Description

```python
import asyncio
import torch
from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine

async def main():
    serve_engine = HiggsAudioServeEngine(
        model_name_or_path="bosonai/higgs-audio-v2-generation-3B-base",
        audio_tokenizer_name_or_path="bosonai/higgs-audio-v2-tokenizer",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    text_chunks = [
        "Hello, welcome to our podcast.",
        "Today we're discussing AI advancements.",
        "Let's dive into the details.",
    ]

    audio_tokens = []

    async for delta in serve_engine.generate_streaming_with_context(
        text_chunks=text_chunks,
        scene_prompt="Audio is recorded from a quiet room.",
        speaker_descriptions="masculine;moderate pitch;professional",
        generation_chunk_buffer_size=5,
        temperature=0.7,
    ):
        if delta.audio_tokens is not None:
            audio_tokens.append(delta.audio_tokens)

asyncio.run(main())
```

### Example 2: Voice Cloning with Reference Audio

```python
async for delta in serve_engine.generate_streaming_with_context(
    text_chunks=text_chunks,
    scene_prompt="Audio is recorded from a quiet room.",
    ref_audio_paths=["path/to/reference_voice.wav"],
    generation_chunk_buffer_size=5,
    temperature=0.3,  # Lower temperature for more faithful cloning
):
    if delta.audio_tokens is not None:
        audio_tokens.append(delta.audio_tokens)
```

### Example 3: Multi-Speaker Dialogue

```python
text_chunks = [
    "[SPEAKER0] Hello! How are you?",
    "[SPEAKER1] I'm doing great, thanks!",
    "[SPEAKER0] What have you been working on?",
    "[SPEAKER1] I've been exploring AI research.",
]

speaker_descriptions = [
    "SPEAKER0: feminine;warm;friendly",
    "SPEAKER1: masculine;deep;calm",
]

async for delta in serve_engine.generate_streaming_with_context(
    text_chunks=text_chunks,
    scene_prompt="Two people having a conversation in a quiet room.",
    speaker_descriptions=speaker_descriptions,
    generation_chunk_buffer_size=5,
    temperature=0.7,
):
    if delta.audio_tokens is not None:
        audio_tokens.append(delta.audio_tokens)
```

### Example 4: Processing Audio Tokens to Waveform

```python
import torch
import soundfile as sf
from boson_multimodal.model.higgs_audio.utils import revert_delay_pattern

# After collecting audio_tokens from streaming
if audio_tokens:
    # Stack tokens
    audio_tensor = torch.stack(audio_tokens, dim=1)

    # Revert delay pattern and clip to valid range
    vq_code = revert_delay_pattern(audio_tensor).clip(
        0, serve_engine.audio_codebook_size - 1
    )[:, 1:-1]

    # Decode to waveform
    waveform = serve_engine.audio_tokenizer.decode(vq_code.unsqueeze(0))[0, 0]

    # Save to file
    sf.write(
        "output.wav",
        waveform,
        serve_engine.audio_tokenizer.sampling_rate
    )
```

## Best Practices

### Buffer Size Configuration

- **Small buffer (1-3 chunks)**: Lower memory usage, slightly less context
- **Medium buffer (5-7 chunks)**: Good balance (recommended)
- **Large buffer (10+ chunks)**: Maximum context, higher memory usage
- **Unlimited buffer (None)**: Not recommended for long generations

### Temperature Settings

- **Voice cloning (0.2-0.4)**: Lower temperature for faithful reproduction
- **Smart voice (0.6-0.8)**: Moderate temperature for natural variation
- **Creative generation (0.9-1.2)**: Higher temperature for diverse outputs

### Text Chunking

Chunk text by:
- **Sentences**: Natural pauses, good for single speaker
- **Paragraphs**: Longer chunks, better for narrative
- **Dialogue turns**: Essential for multi-speaker scenarios

Example:
```python
# Chunk by sentences
text_chunks = text.split('. ')

# Chunk by paragraphs
text_chunks = text.split('\n\n')

# Chunk by speaker turns
text_chunks = [turn for turn in dialogue.split('[SPEAKER') if turn.strip()]
```

## Implementation Details

### Key Components

1. **`StreamingVoiceGenerator`**: Manages rolling context buffer
   - Initializes base messages and reference audio
   - Builds ChatMLDatasetSample with context
   - Processes and accumulates audio tokens
   - Trims buffers to prevent unbounded growth

2. **`generate_streaming_with_context`**: High-level API
   - Creates StreamingVoiceGenerator instance
   - Delegates to generator.generate_streaming()
   - Yields deltas to caller

### Context Buffer Structure

```python
# Base context (immutable after initialization)
base_messages: List[Message]  # System + reference audio messages
ref_audio_ids: List[torch.Tensor]  # Reference audio tokens

# Rolling buffers (updated after each chunk)
generated_audio_ids: List[torch.Tensor]  # Generated audio from previous chunks
generation_messages: List[Message]  # User/assistant pairs from previous chunks
```

### Audio Token Processing Pipeline

```
Raw audio tokens from streamer
    ↓
Stack into tensor [num_codebooks, seq_len]
    ↓
revert_delay_pattern()
    ↓
Clip to [0, codebook_size-1]
    ↓
Trim padding [:, 1:-1]
    ↓
Add to generated_audio_ids buffer
```

## Performance Considerations

### Memory Usage

Memory scales with:
- `generation_chunk_buffer_size`: Larger buffer = more memory
- `num_codebooks`: Model configuration (typically 8)
- Audio length per chunk: Longer chunks = more tokens

Estimate: ~1MB per chunk for typical 5-second audio segments

### Latency

- **First chunk**: Includes model loading + generation
- **Subsequent chunks**: Only generation time
- **Streaming**: Tokens available immediately as generated

### Throughput

- Process multiple requests in parallel by creating separate ServeEngine instances
- Use CUDA graphs (automatically enabled on GPU) for faster generation
- Batch size is currently 1 (multi-batch support planned)

## Troubleshooting

### Voice Not Consistent Across Chunks

1. Increase `generation_chunk_buffer_size` (try 7-10)
2. Use reference audio for voice cloning
3. Lower temperature (0.3-0.5)
4. Ensure speaker descriptions are detailed

### Memory Issues

1. Reduce `generation_chunk_buffer_size`
2. Reduce `max_new_tokens` per chunk
3. Process longer text in smaller chunks
4. Clear buffers between unrelated generations

### Generation Too Slow

1. Use GPU if available
2. Reduce `max_new_tokens`
3. Enable CUDA graphs (automatic on CUDA devices)
4. Consider reducing `ras_win_len` if quality permits

## Examples

See the following example scripts:

- `boson_multimodal/examples/simple_streaming_context.py`: Basic usage
- `boson_multimodal/examples/streaming_with_context_example.py`: Comprehensive examples

Run examples:
```bash
# Simple example
python boson_multimodal/examples/simple_streaming_context.py

# All examples
python boson_multimodal/examples/streaming_with_context_example.py
```

## Technical Background

This implementation is based on the pattern found in `boson_multimodal/examples/generation.py` where the `HiggsAudioModelClient.generate()` method demonstrates chunk-by-chunk generation with rolling context:

```python
# From generation.py - the pattern we implemented
context_audio_ids = audio_ids + generated_audio_ids  # Line 307
generated_audio_ids.append(audio_out_ids)  # Line 360
generation_messages.append(...)  # Lines 362-366
```

Key insight: By accumulating `generated_audio_ids` and `generation_messages` as rolling context, subsequent generations receive information about the voice characteristics and conversation history from previous chunks, enabling consistent voice generation.

## Future Enhancements

Planned features:
- [ ] Multi-batch support for parallel chunk processing
- [ ] Automatic text chunking utilities
- [ ] Voice profile presets
- [ ] Cross-fade audio stitching for seamless transitions
- [ ] Streaming waveform output (PCM chunks)
- [ ] Dynamic buffer size based on memory availability

## License

This feature is part of the boson_multimodal package and follows the same licensing terms.
