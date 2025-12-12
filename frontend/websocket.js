/**
 * websocket.js - WebSocket Connection Management
 * Handles connection, reconnection, health monitoring, and message routing
 */

// ============================================
// STATE
// ============================================
const state = {
  socket: null,
  status: 'disconnected', // 'connecting' | 'connected' | 'disconnected' | 'reconnecting'
  reconnectAttempts: 0,
  lastMessageTime: 0,
  lastPongTime: 0,
  pingInterval: null,
  pongTimeout: null,
  reconnectTimeout: null,
  messageListeners: new Set(),
  audioListeners: new Set(),
  connectionListeners: new Set(),
}

// ============================================
// CONFIGURATION
// ============================================
const config = {
  url: `ws://${window.location.host}/ws`,
  reconnect: true,
  maxReconnectAttempts: 10,
  reconnectBaseDelay: 1000,
  reconnectMaxDelay: 30000,
  pingIntervalMs: 30000,
  pongTimeoutMs: 5000,
}

// ============================================
// CONNECTION MANAGEMENT
// ============================================

/**
 * Connect to WebSocket server
 * @param {string} [url] - Optional custom WebSocket URL
 * @returns {Promise<boolean>} - True if connected successfully
 */
export function connect(url) {
  return new Promise((resolve) => {
    if (state.socket?.readyState === WebSocket.OPEN) {
      resolve(true)
      return
    }

    const wsUrl = url || config.url
    setStatus('connecting')

    try {
      state.socket = new WebSocket(wsUrl)
      state.socket.binaryType = 'arraybuffer'

      state.socket.onopen = () => {
        console.log('[WS] Connected')
        state.reconnectAttempts = 0
        state.lastMessageTime = Date.now()
        setStatus('connected')
        startHeartbeat()
        resolve(true)
      }

      state.socket.onclose = (event) => {
        console.log(`[WS] Closed: code=${event.code}, reason=${event.reason}`)
        stopHeartbeat()
        setStatus('disconnected')

        if (config.reconnect && event.code !== 1000) {
          scheduleReconnect()
        }
      }

      state.socket.onerror = (error) => {
        console.error('[WS] Error:', error)
      }

      // Single persistent message handler - critical for preventing zombie connections
      state.socket.onmessage = handleMessage

    } catch (error) {
      console.error('[WS] Connection failed:', error)
      setStatus('disconnected')
      resolve(false)
    }
  })
}

/**
 * Disconnect from WebSocket server
 */
export function disconnect() {
  config.reconnect = false
  stopHeartbeat()
  clearTimeout(state.reconnectTimeout)

  if (state.socket) {
    state.socket.onclose = null // Prevent reconnection
    state.socket.close(1000, 'Client disconnect')
    state.socket = null
  }

  setStatus('disconnected')
  console.log('[WS] Disconnected')
}

/**
 * Schedule reconnection with exponential backoff
 */
function scheduleReconnect() {
  if (state.reconnectAttempts >= config.maxReconnectAttempts) {
    console.error('[WS] Max reconnection attempts reached')
    setStatus('disconnected')
    return
  }

  const delay = Math.min(
    config.reconnectBaseDelay * Math.pow(2, state.reconnectAttempts),
    config.reconnectMaxDelay
  )

  console.log(`[WS] Reconnecting in ${delay}ms (attempt ${state.reconnectAttempts + 1})`)
  setStatus('reconnecting')

  state.reconnectTimeout = setTimeout(() => {
    state.reconnectAttempts++
    connect()
  }, delay)
}

// ============================================
// HEARTBEAT / PING-PONG
// ============================================

/**
 * Start heartbeat ping interval
 */
function startHeartbeat() {
  stopHeartbeat()

  state.pingInterval = setInterval(() => {
    if (state.socket?.readyState === WebSocket.OPEN) {
      // Send ping
      sendText({ type: 'ping' })

      // Set timeout for pong response
      state.pongTimeout = setTimeout(() => {
        console.warn('[WS] Pong timeout - connection may be dead')
        // Force reconnection
        if (state.socket) {
          state.socket.close(4000, 'Pong timeout')
        }
      }, config.pongTimeoutMs)
    }
  }, config.pingIntervalMs)
}

/**
 * Stop heartbeat ping interval
 */
function stopHeartbeat() {
  clearInterval(state.pingInterval)
  clearTimeout(state.pongTimeout)
  state.pingInterval = null
  state.pongTimeout = null
}

/**
 * Handle pong response
 */
function handlePong() {
  clearTimeout(state.pongTimeout)
  state.lastPongTime = Date.now()
}

// ============================================
// MESSAGE HANDLING
// ============================================

/**
 * Handle incoming WebSocket message
 * @param {MessageEvent} event
 */
function handleMessage(event) {
  state.lastMessageTime = Date.now()

  if (event.data instanceof ArrayBuffer) {
    // Binary message = TTS audio
    notifyAudioListeners(event.data)
  } else {
    // Text message = JSON
    try {
      const message = JSON.parse(event.data)

      // Handle pong internally
      if (message.type === 'pong') {
        handlePong()
        return
      }

      notifyMessageListeners(message)
    } catch (error) {
      console.error('[WS] Failed to parse message:', error)
    }
  }
}

/**
 * Send JSON message to server
 * @param {object} message - Message object to send
 */
export function sendText(message) {
  if (state.socket?.readyState === WebSocket.OPEN) {
    state.socket.send(JSON.stringify(message))
  } else {
    console.warn('[WS] Cannot send - not connected')
  }
}

/**
 * Send binary audio data to server
 * @param {ArrayBuffer} audioData - PCM16 audio data
 */
export function sendAudio(audioData) {
  if (state.socket?.readyState === WebSocket.OPEN) {
    state.socket.send(audioData)
  }
}

// ============================================
// EVENT LISTENERS
// ============================================

/**
 * Register message listener
 * @param {Function} handler - Callback(message: object)
 * @returns {Function} - Unsubscribe function
 */
export function onMessage(handler) {
  state.messageListeners.add(handler)
  return () => state.messageListeners.delete(handler)
}

/**
 * Register audio listener
 * @param {Function} handler - Callback(audioData: ArrayBuffer)
 * @returns {Function} - Unsubscribe function
 */
export function onAudio(handler) {
  state.audioListeners.add(handler)
  return () => state.audioListeners.delete(handler)
}

/**
 * Register connection state listener
 * @param {Function} handler - Callback(status: string)
 * @returns {Function} - Unsubscribe function
 */
export function onConnectionChange(handler) {
  state.connectionListeners.add(handler)
  return () => state.connectionListeners.delete(handler)
}

/**
 * Notify all message listeners
 * @param {object} message
 */
function notifyMessageListeners(message) {
  state.messageListeners.forEach(handler => {
    try {
      handler(message)
    } catch (error) {
      console.error('[WS] Message handler error:', error)
    }
  })
}

/**
 * Notify all audio listeners
 * @param {ArrayBuffer} audioData
 */
function notifyAudioListeners(audioData) {
  state.audioListeners.forEach(handler => {
    try {
      handler(audioData)
    } catch (error) {
      console.error('[WS] Audio handler error:', error)
    }
  })
}

/**
 * Update connection status and notify listeners
 * @param {string} status
 */
function setStatus(status) {
  state.status = status
  state.connectionListeners.forEach(handler => {
    try {
      handler(status)
    } catch (error) {
      console.error('[WS] Connection handler error:', error)
    }
  })
}

// ============================================
// UTILITY EXPORTS
// ============================================

/**
 * Check if WebSocket is connected
 * @returns {boolean}
 */
export function isConnected() {
  return state.socket?.readyState === WebSocket.OPEN
}

/**
 * Get current connection status
 * @returns {string}
 */
export function getStatus() {
  return state.status
}

/**
 * Get connection state details
 * @returns {object}
 */
export function getState() {
  return {
    status: state.status,
    connected: isConnected(),
    reconnectAttempts: state.reconnectAttempts,
    lastMessageTime: state.lastMessageTime,
    lastPongTime: state.lastPongTime,
  }
}

// ============================================
// CONVENIENCE METHODS FOR COMMON MESSAGES
// ============================================

/**
 * Send user text message
 * @param {string} text
 */
export function sendUserMessage(text) {
  sendText({ type: 'user_message', data: { text } })
}

/**
 * Start STT listening
 */
export function startListening() {
  sendText({ type: 'start_listening' })
}

/**
 * Stop STT listening
 */
export function stopListening() {
  sendText({ type: 'stop_listening' })
}

/**
 * Send interrupt signal
 */
export function sendInterrupt() {
  sendText({ type: 'interrupt' })
}

/**
 * Clear conversation history
 */
export function clearHistory() {
  sendText({ type: 'clear_history' })
}

/**
 * Refresh active characters from database
 */
export function refreshActiveCharacters() {
  sendText({ type: 'refresh_active_characters' })
}

/**
 * Update model settings
 * @param {object} settings
 */
export function updateModelSettings(settings) {
  sendText({ type: 'model_settings', data: settings })
}

/**
 * Set active characters
 * @param {Array} characters
 */
export function setCharacters(characters) {
  sendText({ type: 'set_characters', data: { characters } })
}
