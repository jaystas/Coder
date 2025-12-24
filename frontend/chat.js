/**
 * chat.js - Chat UI and Conversation Management
 * Handles message display, streaming responses, and STT preview
 */

import * as websocket from './websocket.js'
import { characterCache } from './characterCache.js'

// ============================================
// STATE
// ============================================
const state = {
  conversation: [],
  currentStreamElement: null,
  currentStreamText: '',
  currentMessageId: null,
  isStreaming: false,
  activeCharacters: new Map(), // id -> character object
}

// ============================================
// INITIALIZATION
// ============================================

/**
 * Initialize chat system
 */
export function initChat() {
  // Subscribe to WebSocket messages
  websocket.onMessage(handleServerMessage)

  // Load active characters for avatar display
  loadActiveCharacters()

  console.log('[Chat] Initialized')
}

/**
 * Load active characters from cache
 */
async function loadActiveCharacters() {
  try {
    if (!characterCache.isInitialized) {
      await characterCache.initialize()
    }

    const characters = characterCache.getAllCharacters()
    characters.forEach(char => {
      if (char.is_active) {
        state.activeCharacters.set(char.id, char)
        state.activeCharacters.set(char.name, char) // Also index by name for lookup
      }
    })

    console.log(`[Chat] Loaded ${state.activeCharacters.size / 2} active characters`)
  } catch (error) {
    console.error('[Chat] Failed to load characters:', error)
  }
}

// ============================================
// MESSAGE HANDLING
// ============================================

/**
 * Handle incoming server messages
 * @param {object} message
 */
function handleServerMessage(message) {
  switch (message.type) {
    case 'stt_update':
      updateSTTPreview(message.text, false)
      break

    case 'stt_stabilized':
      updateSTTPreview(message.text, true)
      break

    case 'stt_final':
      finalizeUserMessage(message.text)
      break

    case 'text_chunk':
      handleResponseChunk(message.data)
      break

    case 'audio_complete':
      handleAudioComplete(message)
      break
  }
}

/**
 * Handle LLM response chunk
 * @param {object} data - {text, character_name, message_id, is_final}
 */
function handleResponseChunk(data) {
  const { text, character_name, message_id, is_final } = data

  // Start new message if needed
  if (!state.isStreaming || state.currentMessageId !== message_id) {
    // Get character info for avatar
    const character = state.activeCharacters.get(character_name) || null

    state.currentStreamElement = createAssistantMessageElement(character_name, character)
    state.currentStreamText = ''
    state.currentMessageId = message_id
    state.isStreaming = true
  }

  // Append text chunk
  if (text) {
    state.currentStreamText += text
    renderMessageContent(state.currentStreamElement, state.currentStreamText)
  }

  // Finalize message
  if (is_final) {
    state.conversation.push({
      role: 'assistant',
      name: character_name,
      content: state.currentStreamText,
      timestamp: Date.now(),
    })

    state.isStreaming = false
    state.currentStreamElement = null
    state.currentMessageId = null
    scrollToBottom()
  }
}

/**
 * Handle audio complete notification
 * @param {object} data - {message_id, character_id}
 */
function handleAudioComplete(data) {
  const { message_id, character_id } = data
  console.log(`[Chat] Audio complete for message ${message_id} from character ${character_id}`)
  // Optional: Add visual indicator that audio is ready/complete
}

// ============================================
// STT PREVIEW
// ============================================

/**
 * Update STT preview display
 * @param {string} text - Transcribed text
 * @param {boolean} isStabilized - Whether text is stabilized
 */
function updateSTTPreview(text, isStabilized) {
  let preview = document.getElementById('stt-preview')

  if (!preview) {
    preview = createSTTPreviewElement()
  }

  preview.textContent = text
  preview.classList.toggle('stabilized', isStabilized)
}

/**
 * Create STT preview element
 * @returns {HTMLElement}
 */
function createSTTPreviewElement() {
  const messagesArea = getMessagesArea()
  if (!messagesArea) return null

  const preview = document.createElement('div')
  preview.id = 'stt-preview'
  preview.className = 'stt-preview'
  messagesArea.appendChild(preview)

  scrollToBottom()
  return preview
}

/**
 * Remove STT preview element
 */
function removeSTTPreview() {
  const preview = document.getElementById('stt-preview')
  if (preview) {
    preview.remove()
  }
}

/**
 * Finalize user message from STT
 * @param {string} text
 */
function finalizeUserMessage(text) {
  removeSTTPreview()

  if (text && text.trim()) {
    addUserMessage(text)
  }
}

// ============================================
// MESSAGE CREATION
// ============================================

/**
 * Add user message to chat
 * @param {string} text
 */
export function addUserMessage(text) {
  const messagesArea = getMessagesArea()
  if (!messagesArea) return

  const messageEl = document.createElement('div')
  messageEl.className = 'chat-message user'
  messageEl.innerHTML = `
    <div class="message-bubble">
      <div class="message-content">${escapeHtml(text)}</div>
      <div class="message-time">${formatTime(new Date())}</div>
    </div>
  `

  messagesArea.appendChild(messageEl)
  scrollToBottom()

  // Add to conversation history
  state.conversation.push({
    role: 'user',
    content: text,
    timestamp: Date.now(),
  })
}

/**
 * Create assistant message element
 * @param {string} characterName
 * @param {object|null} character - Character object with avatar
 * @returns {HTMLElement} - Content element for streaming
 */
function createAssistantMessageElement(characterName, character) {
  const messagesArea = getMessagesArea()
  if (!messagesArea) return null

  // Build avatar HTML
  let avatarHtml
  if (character?.image_url) {
    avatarHtml = `<img src="${escapeHtml(character.image_url)}" alt="${escapeHtml(characterName)}" />`
  } else {
    // Default avatar with initials
    const initials = characterName.split(' ').map(n => n[0]).join('').substring(0, 2).toUpperCase()
    avatarHtml = `<span class="avatar-initials">${initials}</span>`
  }

  const messageEl = document.createElement('div')
  messageEl.className = 'chat-message assistant'
  messageEl.innerHTML = `
    <div class="message-avatar">
      ${avatarHtml}
    </div>
    <div class="message-body">
      <div class="message-header">
        <span class="character-name">${escapeHtml(characterName)}</span>
        <span class="message-time">${formatTime(new Date())}</span>
      </div>
      <div class="message-bubble">
        <div class="message-content"></div>
      </div>
    </div>
  `

  messagesArea.appendChild(messageEl)
  scrollToBottom()

  return messageEl.querySelector('.message-content')
}

/**
 * Render message content (supports basic markdown)
 * @param {HTMLElement} element
 * @param {string} text
 */
function renderMessageContent(element, text) {
  if (!element) return

  // Basic markdown-like rendering
  let html = escapeHtml(text)

  // Convert newlines to <br>
  html = html.replace(/\n/g, '<br>')

  // Bold **text**
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')

  // Italic *text*
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>')

  // Code `text`
  html = html.replace(/`(.+?)`/g, '<code>$1</code>')

  element.innerHTML = html
}

// ============================================
// PUBLIC API
// ============================================

/**
 * Send message to server
 * @param {string} text
 */
export function sendMessage(text) {
  if (!text || !text.trim()) return

  addUserMessage(text)
  websocket.sendUserMessage(text)
}

/**
 * Clear chat messages
 */
export function clearChat() {
  const messagesArea = getMessagesArea()
  if (messagesArea) {
    messagesArea.innerHTML = ''
  }

  state.conversation = []
  state.isStreaming = false
  state.currentStreamElement = null

  // Also clear server-side history
  websocket.clearHistory()
}

/**
 * Get current conversation
 * @returns {Array}
 */
export function getConversation() {
  return [...state.conversation]
}

/**
 * Set typing indicator
 * @param {string} characterName
 * @param {boolean} show
 */
export function setTypingIndicator(characterName, show) {
  let indicator = document.getElementById('typing-indicator')

  if (show) {
    if (!indicator) {
      indicator = createTypingIndicator(characterName)
    }
  } else {
    if (indicator) {
      indicator.remove()
    }
  }
}

/**
 * Create typing indicator element
 * @param {string} characterName
 * @returns {HTMLElement}
 */
function createTypingIndicator(characterName) {
  const messagesArea = getMessagesArea()
  if (!messagesArea) return null

  const indicator = document.createElement('div')
  indicator.id = 'typing-indicator'
  indicator.className = 'typing-indicator'
  indicator.innerHTML = `
    <span class="typing-name">${escapeHtml(characterName)}</span>
    <span class="typing-text">is typing</span>
    <div class="typing-dots">
      <span></span>
      <span></span>
      <span></span>
    </div>
  `

  messagesArea.appendChild(indicator)
  scrollToBottom()

  return indicator
}

// ============================================
// UTILITY FUNCTIONS
// ============================================

/**
 * Get messages area element
 * @returns {HTMLElement|null}
 */
function getMessagesArea() {
  return document.querySelector('.messages-area')
}

/**
 * Scroll messages area to bottom
 */
function scrollToBottom() {
  const messagesArea = getMessagesArea()
  if (messagesArea) {
    messagesArea.scrollTop = messagesArea.scrollHeight
  }
}

/**
 * Escape HTML special characters
 * @param {string} text
 * @returns {string}
 */
function escapeHtml(text) {
  const div = document.createElement('div')
  div.textContent = text
  return div.innerHTML
}

/**
 * Format time for display
 * @param {Date} date
 * @returns {string}
 */
function formatTime(date) {
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

/**
 * Refresh active characters (call after character changes)
 */
export function refreshCharacters() {
  state.activeCharacters.clear()
  loadActiveCharacters()
}
