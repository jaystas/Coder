/**
 * stt-audio.js - Audio Input Capture for STT
 * Handles microphone capture, VAD, and audio streaming to server
 */

import * as websocket from './websocket.js'

// ============================================
// STATE
// ============================================
const state = {
  audioContext: null,
  mediaStream: null,
  workletNode: null,
  sourceNode: null,
  status: 'idle', // 'idle' | 'listening' | 'recording' | 'paused'
  isTTSPlaying: false,
  stateListeners: new Set(),
}

// ============================================
// CONFIGURATION
// ============================================
const config = {
  sampleRate: 16000,
  channelCount: 1,
  bufferSize: 4096,
}

// ============================================
// INITIALIZATION
// ============================================

/**
 * Initialize audio capture system
 * @returns {Promise<boolean>} - True if initialized successfully
 */
export async function initAudioCapture() {
  try {
    // Request microphone access
    state.mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        sampleRate: config.sampleRate,
        channelCount: config.channelCount,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      }
    })

    // Create audio context
    state.audioContext = new AudioContext({
      sampleRate: config.sampleRate,
    })

    // Load AudioWorklet processor
    const processorUrl = new URL('./stt-processor.js', import.meta.url)
    await state.audioContext.audioWorklet.addModule(processorUrl)

    // Create worklet node
    state.workletNode = new AudioWorkletNode(
      state.audioContext,
      'stt-processor',
      {
        processorOptions: {
          bufferSize: config.bufferSize,
        }
      }
    )

    // Connect audio pipeline
    state.sourceNode = state.audioContext.createMediaStreamSource(state.mediaStream)
    state.sourceNode.connect(state.workletNode)

    // Handle messages from worklet
    state.workletNode.port.onmessage = handleWorkletMessage

    console.log('[STT] Audio capture initialized')
    return true

  } catch (error) {
    console.error('[STT] Failed to initialize audio capture:', error)
    return false
  }
}

/**
 * Check if audio is initialized
 * @returns {boolean}
 */
export function isInitialized() {
  return state.audioContext !== null && state.workletNode !== null
}

// ============================================
// RECORDING CONTROL
// ============================================

/**
 * Start recording/listening
 */
export function startRecording() {
  if (!isInitialized()) {
    console.warn('[STT] Not initialized')
    return
  }

  if (state.status === 'recording' || state.status === 'listening') {
    return
  }

  // Resume audio context if suspended
  if (state.audioContext.state === 'suspended') {
    state.audioContext.resume()
  }

  // If TTS is playing, go to paused state instead
  if (state.isTTSPlaying) {
    setStatus('paused')
    return
  }

  setStatus('listening')
  state.workletNode.port.postMessage({ command: 'start' })
  websocket.startListening()

  console.log('[STT] Started listening')
}

/**
 * Stop recording/listening
 */
export function stopRecording() {
  if (!isInitialized()) return

  setStatus('idle')
  state.workletNode.port.postMessage({ command: 'stop' })
  websocket.stopListening()

  console.log('[STT] Stopped')
}

/**
 * Check if currently recording
 * @returns {boolean}
 */
export function isRecording() {
  return state.status === 'recording'
}

/**
 * Check if currently listening (waiting for voice)
 * @returns {boolean}
 */
export function isListening() {
  return state.status === 'listening'
}

/**
 * Check if active (listening or recording)
 * @returns {boolean}
 */
export function isActive() {
  return state.status === 'listening' || state.status === 'recording'
}

/**
 * Get current status
 * @returns {string}
 */
export function getStatus() {
  return state.status
}

// ============================================
// TTS COORDINATION
// ============================================

/**
 * Set TTS playing state (called by tts-audio.js)
 * @param {boolean} isPlaying
 */
export function setTTSPlaying(isPlaying) {
  state.isTTSPlaying = isPlaying

  if (isPlaying) {
    // Pause recording during TTS
    if (state.status === 'recording' || state.status === 'listening') {
      state.workletNode?.port.postMessage({ command: 'pause' })
      setStatus('paused')
      console.log('[STT] Paused for TTS playback')
    }
  } else {
    // Resume listening after TTS ends
    if (state.status === 'paused') {
      state.workletNode?.port.postMessage({ command: 'resume' })
      setStatus('listening')
      console.log('[STT] Resumed after TTS playback')
    }
  }
}

// ============================================
// WORKLET MESSAGE HANDLING
// ============================================

/**
 * Handle messages from AudioWorklet processor
 * @param {MessageEvent} event
 */
function handleWorkletMessage(event) {
  const { type, data } = event.data

  switch (type) {
    case 'audio':
      // data is Int16Array - send to server
      if (state.status === 'recording') {
        websocket.sendAudio(data.buffer)
      }
      break

    case 'vad_start':
      // Voice activity detected
      if (state.status === 'listening') {
        setStatus('recording')
        console.log('[STT] Voice detected - recording')
      }
      break

    case 'vad_stop':
      // Silence detected after speech
      if (state.status === 'recording') {
        setStatus('listening')
        console.log('[STT] Silence detected - listening')
      }
      break
  }
}

// ============================================
// STATE MANAGEMENT
// ============================================

/**
 * Set status and notify listeners
 * @param {string} newStatus
 */
function setStatus(newStatus) {
  const oldStatus = state.status
  state.status = newStatus

  if (oldStatus !== newStatus) {
    notifyStateListeners(newStatus, oldStatus)
  }
}

/**
 * Register state change listener
 * @param {Function} handler - Callback(newStatus, oldStatus)
 * @returns {Function} - Unsubscribe function
 */
export function onStateChange(handler) {
  state.stateListeners.add(handler)
  return () => state.stateListeners.delete(handler)
}

/**
 * Notify all state listeners
 * @param {string} newStatus
 * @param {string} oldStatus
 */
function notifyStateListeners(newStatus, oldStatus) {
  state.stateListeners.forEach(handler => {
    try {
      handler(newStatus, oldStatus)
    } catch (error) {
      console.error('[STT] State listener error:', error)
    }
  })
}

// ============================================
// CLEANUP
// ============================================

/**
 * Cleanup audio resources
 */
export function cleanup() {
  stopRecording()

  if (state.sourceNode) {
    state.sourceNode.disconnect()
    state.sourceNode = null
  }

  if (state.workletNode) {
    state.workletNode.disconnect()
    state.workletNode = null
  }

  if (state.mediaStream) {
    state.mediaStream.getTracks().forEach(track => track.stop())
    state.mediaStream = null
  }

  if (state.audioContext) {
    state.audioContext.close()
    state.audioContext = null
  }

  console.log('[STT] Cleaned up')
}
