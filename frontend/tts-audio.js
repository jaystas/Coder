/**
 * tts-audio.js - TTS Audio Playback
 * Handles audio queue, playback, and coordination with STT
 */

import * as websocket from './websocket.js'
import * as sttAudio from './stt-audio.js'

// ============================================
// STATE
// ============================================
const state = {
  audioContext: null,
  gainNode: null,
  audioQueue: [],
  currentSource: null,
  isPlaying: false,
  volume: 1.0,
  playbackListeners: new Set(),
}

// ============================================
// CONFIGURATION
// ============================================
const config = {
  sampleRate: 24000,       // TTS output sample rate from server
  channelCount: 1,         // Mono
  minBufferMs: 50,         // Start playback after this much buffered
}

// ============================================
// INITIALIZATION
// ============================================

/**
 * Initialize TTS playback system
 * @returns {Promise<void>}
 */
export async function initTTSPlayback() {
  state.audioContext = new AudioContext({
    sampleRate: config.sampleRate,
  })

  // Create gain node for volume control
  state.gainNode = state.audioContext.createGain()
  state.gainNode.connect(state.audioContext.destination)
  state.gainNode.gain.value = state.volume

  // Subscribe to WebSocket audio
  websocket.onAudio(queueAudioChunk)

  // Resume audio context on user interaction (browser policy)
  const resumeContext = () => {
    if (state.audioContext?.state === 'suspended') {
      state.audioContext.resume()
    }
    document.removeEventListener('click', resumeContext)
    document.removeEventListener('keydown', resumeContext)
  }

  document.addEventListener('click', resumeContext)
  document.addEventListener('keydown', resumeContext)

  console.log('[TTS] Playback initialized')
}

/**
 * Check if TTS is initialized
 * @returns {boolean}
 */
export function isInitialized() {
  return state.audioContext !== null
}

// ============================================
// AUDIO QUEUE
// ============================================

/**
 * Queue audio chunk for playback
 * @param {ArrayBuffer} pcm16Data - PCM16 audio data
 */
export function queueAudioChunk(pcm16Data) {
  if (!state.audioContext) {
    console.warn('[TTS] Not initialized')
    return
  }

  // Convert PCM16 to AudioBuffer
  const int16Array = new Int16Array(pcm16Data)
  const float32Array = int16ToFloat32(int16Array)

  const audioBuffer = state.audioContext.createBuffer(
    config.channelCount,
    float32Array.length,
    config.sampleRate
  )
  audioBuffer.getChannelData(0).set(float32Array)

  state.audioQueue.push(audioBuffer)

  // Auto-start playback if not playing
  if (!state.isPlaying) {
    const bufferedMs = getBufferedDuration() * 1000
    if (bufferedMs >= config.minBufferMs) {
      play()
    }
  }
}

/**
 * Get total buffered duration in seconds
 * @returns {number}
 */
function getBufferedDuration() {
  return state.audioQueue.reduce((sum, buf) => sum + buf.duration, 0)
}

// ============================================
// PLAYBACK CONTROL
// ============================================

/**
 * Start playback
 */
export function play() {
  if (state.audioQueue.length === 0) {
    return
  }

  // Resume audio context if suspended
  if (state.audioContext?.state === 'suspended') {
    state.audioContext.resume()
  }

  if (!state.isPlaying) {
    state.isPlaying = true
    notifyPlaybackChange(true)

    // Notify STT to pause recording
    sttAudio.setTTSPlaying(true)
  }

  playNextChunk()
}

/**
 * Play next chunk from queue
 */
function playNextChunk() {
  if (state.audioQueue.length === 0) {
    // Queue empty - playback complete
    state.isPlaying = false
    state.currentSource = null
    notifyPlaybackChange(false)

    // Notify STT to resume recording
    sttAudio.setTTSPlaying(false)
    return
  }

  const audioBuffer = state.audioQueue.shift()

  state.currentSource = state.audioContext.createBufferSource()
  state.currentSource.buffer = audioBuffer
  state.currentSource.connect(state.gainNode)

  state.currentSource.onended = () => {
    if (state.isPlaying) {
      playNextChunk()
    }
  }

  state.currentSource.start()
}

/**
 * Stop playback immediately (for interrupt)
 */
export function stop() {
  // Clear queue
  state.audioQueue = []

  // Stop current playback
  if (state.currentSource) {
    state.currentSource.onended = null

    try {
      state.currentSource.stop()
    } catch (e) {
      // Ignore if already stopped
    }

    state.currentSource = null
  }

  const wasPlaying = state.isPlaying
  state.isPlaying = false

  if (wasPlaying) {
    notifyPlaybackChange(false)

    // Notify STT to resume recording
    sttAudio.setTTSPlaying(false)

    // Send interrupt to server to stop generating more audio
    websocket.sendInterrupt()

    console.log('[TTS] Playback stopped (interrupt)')
  }
}

/**
 * Pause playback
 */
export function pause() {
  if (state.currentSource) {
    state.currentSource.onended = null

    try {
      state.currentSource.stop()
    } catch (e) {
      // Ignore
    }

    state.currentSource = null
  }

  // Keep isPlaying true so we know to resume
  // Audio in queue is preserved
}

/**
 * Resume playback
 */
export function resume() {
  if (state.isPlaying && state.audioQueue.length > 0) {
    playNextChunk()
  }
}

/**
 * Check if currently playing
 * @returns {boolean}
 */
export function isPlaying() {
  return state.isPlaying
}

// ============================================
// VOLUME CONTROL
// ============================================

/**
 * Set playback volume
 * @param {number} volume - 0.0 to 1.0
 */
export function setVolume(volume) {
  state.volume = Math.max(0, Math.min(1, volume))

  if (state.gainNode) {
    state.gainNode.gain.value = state.volume
  }
}

/**
 * Get current volume
 * @returns {number}
 */
export function getVolume() {
  return state.volume
}

// ============================================
// EVENT LISTENERS
// ============================================

/**
 * Register playback state listener
 * @param {Function} handler - Callback(isPlaying: boolean)
 * @returns {Function} - Unsubscribe function
 */
export function onPlaybackChange(handler) {
  state.playbackListeners.add(handler)
  return () => state.playbackListeners.delete(handler)
}

/**
 * Notify all playback listeners
 * @param {boolean} isPlaying
 */
function notifyPlaybackChange(isPlaying) {
  state.playbackListeners.forEach(handler => {
    try {
      handler(isPlaying)
    } catch (error) {
      console.error('[TTS] Playback listener error:', error)
    }
  })
}

// ============================================
// UTILITY
// ============================================

/**
 * Convert Int16Array to Float32Array
 * @param {Int16Array} int16Array
 * @returns {Float32Array}
 */
function int16ToFloat32(int16Array) {
  const float32Array = new Float32Array(int16Array.length)

  for (let i = 0; i < int16Array.length; i++) {
    float32Array[i] = int16Array[i] / 32768.0
  }

  return float32Array
}

/**
 * Clear audio queue without stopping
 */
export function clearQueue() {
  state.audioQueue = []
}

/**
 * Get queue status
 * @returns {object}
 */
export function getStatus() {
  return {
    isPlaying: state.isPlaying,
    queueLength: state.audioQueue.length,
    bufferedDuration: getBufferedDuration(),
    volume: state.volume,
  }
}

// ============================================
// CLEANUP
// ============================================

/**
 * Cleanup TTS resources
 */
export function cleanup() {
  stop()

  if (state.audioContext) {
    state.audioContext.close()
    state.audioContext = null
  }

  state.gainNode = null
  console.log('[TTS] Cleaned up')
}
