/**
 * stt-processor.js - AudioWorkletProcessor for STT
 * Runs in a separate audio thread for low-latency processing
 *
 * Note: This file runs in an AudioWorklet context and cannot import other modules
 */

class STTProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super()

    // Buffer configuration
    this.bufferSize = options.processorOptions?.bufferSize || 4096
    this.buffer = new Float32Array(this.bufferSize)
    this.bufferIndex = 0

    // State
    this.isActive = false
    this.isPaused = false

    // VAD (Voice Activity Detection) configuration
    this.vadThreshold = 0.015          // RMS threshold for voice detection
    this.silenceFrameThreshold = 50    // Frames of silence before vad_stop (~700ms at 128 samples/frame)
    this.silenceFrameCount = 0
    this.isSpeaking = false
    this.hasSpoken = false             // Track if user has spoken during this session

    // Handle commands from main thread
    this.port.onmessage = this.handleCommand.bind(this)
  }

  /**
   * Handle commands from main thread
   * @param {MessageEvent} event
   */
  handleCommand(event) {
    const { command } = event.data

    switch (command) {
      case 'start':
        this.isActive = true
        this.isPaused = false
        this.hasSpoken = false
        this.silenceFrameCount = 0
        break

      case 'stop':
        this.isActive = false
        this.isPaused = false
        this.isSpeaking = false
        this.hasSpoken = false
        break

      case 'pause':
        this.isPaused = true
        break

      case 'resume':
        this.isPaused = false
        break
    }
  }

  /**
   * Process audio samples
   * Called by the audio rendering thread for each quantum of audio
   * @param {Array<Float32Array[]>} inputs - Input audio buffers
   * @param {Array<Float32Array[]>} outputs - Output audio buffers (not used)
   * @param {Record<string, Float32Array>} parameters - Parameter values
   * @returns {boolean} - Return true to keep processor alive
   */
  process(inputs, outputs, parameters) {
    const input = inputs[0]

    // No input or not active
    if (!input || !input[0] || !this.isActive || this.isPaused) {
      return true
    }

    const samples = input[0]

    // Calculate RMS (Root Mean Square) for VAD
    const rms = this.calculateRMS(samples)

    // Voice Activity Detection
    this.processVAD(rms)

    // Buffer samples for transmission
    this.bufferSamples(samples)

    return true // Keep processor alive
  }

  /**
   * Calculate RMS of samples
   * @param {Float32Array} samples
   * @returns {number}
   */
  calculateRMS(samples) {
    let sumSquares = 0
    for (let i = 0; i < samples.length; i++) {
      sumSquares += samples[i] * samples[i]
    }
    return Math.sqrt(sumSquares / samples.length)
  }

  /**
   * Process Voice Activity Detection
   * @param {number} rms - Current RMS level
   */
  processVAD(rms) {
    const wasSpeaking = this.isSpeaking

    if (rms > this.vadThreshold) {
      // Voice detected
      this.isSpeaking = true
      this.silenceFrameCount = 0

      if (!wasSpeaking) {
        this.hasSpoken = true
        this.port.postMessage({ type: 'vad_start' })
      }
    } else if (this.isSpeaking) {
      // Silence during speech
      this.silenceFrameCount++

      if (this.silenceFrameCount > this.silenceFrameThreshold) {
        // Enough silence to consider speech ended
        this.isSpeaking = false
        this.port.postMessage({ type: 'vad_stop' })
      }
    }
  }

  /**
   * Buffer samples and send when full
   * @param {Float32Array} samples
   */
  bufferSamples(samples) {
    for (let i = 0; i < samples.length; i++) {
      this.buffer[this.bufferIndex++] = samples[i]

      if (this.bufferIndex >= this.bufferSize) {
        // Buffer full - convert and send
        const int16Buffer = this.float32ToInt16(this.buffer)

        // Transfer ownership for zero-copy
        this.port.postMessage(
          { type: 'audio', data: int16Buffer },
          [int16Buffer.buffer]
        )

        // Reset buffer
        this.buffer = new Float32Array(this.bufferSize)
        this.bufferIndex = 0
      }
    }
  }

  /**
   * Convert Float32Array to Int16Array (PCM16)
   * @param {Float32Array} float32Array
   * @returns {Int16Array}
   */
  float32ToInt16(float32Array) {
    const int16Array = new Int16Array(float32Array.length)

    for (let i = 0; i < float32Array.length; i++) {
      // Clamp to [-1, 1] range
      const sample = Math.max(-1, Math.min(1, float32Array[i]))

      // Convert to Int16 range [-32768, 32767]
      int16Array[i] = sample < 0
        ? sample * 0x8000
        : sample * 0x7FFF
    }

    return int16Array
  }
}

// Register the processor
registerProcessor('stt-processor', STTProcessor)
