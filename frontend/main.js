/**
 * main.js - Core functionality for aiChat interface
 * Handles navigation and sidebar collapse
 */

// Import editor functions
import { initEditor, handleMic, handleSend } from './editor.js';

// Import characters functions
import { initCharacters } from './characters.js';

// Import chat system modules
import * as websocket from './websocket.js';
import * as chat from './chat.js';
import * as ttsAudio from './tts-audio.js';

// Make functions globally accessible for inline event handlers
window.handleMic = handleMic;
window.handleSend = handleSend;

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', () => {
  initSidebar();
  initNavigation();
});

/**
 * Initialize sidebar collapse functionality
 */
function initSidebar() {
  const sidebar = document.querySelector('.sidebar');
  const collapseBtn = document.querySelector('.collapse-btn');

  if (!sidebar || !collapseBtn) {
    console.warn('Sidebar elements not found');
    return;
  }

  // Load saved sidebar state from localStorage
  const sidebarState = localStorage.getItem('sidebarCollapsed');
  if (sidebarState === 'true') {
    sidebar.classList.add('collapsed');
  }

  // Toggle sidebar on button click
  collapseBtn.addEventListener('click', () => {
    sidebar.classList.toggle('collapsed');

    // Save state to localStorage
    const isCollapsed = sidebar.classList.contains('collapsed');
    localStorage.setItem('sidebarCollapsed', isCollapsed);
  });
}

/**
 * Initialize navigation functionality
 */
function initNavigation() {
  const navLinks = document.querySelectorAll('.nav-link');
  const contentArea = document.querySelector('.content-area');

  if (!contentArea) {
    console.warn('Content area not found');
    return;
  }

  // Handle navigation link clicks
  navLinks.forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();

      // Remove active class from all links
      navLinks.forEach(l => l.classList.remove('active'));

      // Add active class to clicked link
      link.classList.add('active');

      // Get the page name from href (e.g., #home -> home)
      const page = link.getAttribute('href').substring(1);

      // Update the content area
      loadPage(page, contentArea);

      // Update URL hash without scrolling
      history.pushState(null, null, `#${page}`);
    });
  });

  // Handle browser back/forward buttons
  window.addEventListener('popstate', () => {
    const hash = window.location.hash.substring(1) || 'home';
    const activeLink = document.querySelector(`.nav-link[href="#${hash}"]`);

    if (activeLink) {
      navLinks.forEach(l => l.classList.remove('active'));
      activeLink.classList.add('active');
      loadPage(hash, contentArea);
    }
  });

  // Load initial page based on URL hash
  const initialHash = window.location.hash.substring(1) || 'home';
  const initialLink = document.querySelector(`.nav-link[href="#${initialHash}"]`);

  if (initialLink) {
    navLinks.forEach(l => l.classList.remove('active'));
    initialLink.classList.add('active');
    loadPage(initialHash, contentArea);
  } else {
    // Default to first link if hash doesn't match
    const firstLink = navLinks[0];
    if (firstLink) {
      firstLink.classList.add('active');
      const defaultPage = firstLink.getAttribute('href').substring(1);
      loadPage(defaultPage, contentArea);
    }
  }
}

/**
 * Load page content
 * @param {string} page - The page name to load
 * @param {HTMLElement} container - The container element to update
 */
function loadPage(page, container) {
  // Page content templates
  const pageContent = {
    home: `
      <div class="page-content">
        <!-- Info Column (Left) -->
        <div class="info-column">
        </div>

        <!-- Chat Column (Center) -->
        <div class="chat-column">
          <!-- Messages Area -->
          <div class="messages-area">
          </div>

          <!-- Editor Area -->
          <div class="editor-area">
            <div class="editor-container">
              <div class="toolbar" id="toolbar">
                <!-- Toolbar buttons will be dynamically added by editor.js -->
              </div>

              <div class="editor-content">
                <div id="editor"></div>
              </div>

              <div class="editor-footer">
                <button class="mic-button" onclick="handleMic()">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
                    <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
                    <line x1="12" y1="19" x2="12" y2="23"></line>
                    <line x1="8" y1="23" x2="16" y2="23"></line>
                  </svg>
                </button>
                <button class="send-button" onclick="handleSend()">Send</button>
              </div>
            </div>
          </div>
        </div>

        <!-- Settings Column (Right) -->
        <div class="settings-column">
        </div>
      </div>

      <!-- Info Drawer Toggle Button (Left) -->
      <button class="info-drawer-toggle" id="info-drawer-toggle">
        <img class="right-arrow" src="assets/arrow-right.png" alt="arrow-compact" />
      </button>

      <!-- Info Drawer (Left Side) -->
      <div class="info-drawer" id="info-drawer">
        <div class="info-drawer-header">
          <h2>Info Panel</h2>
        </div>
        <div class="info-drawer-content">
          <!-- Your info drawer content goes here -->
          <div class="info-placeholder">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
              <circle cx="12" cy="12" r="10"></circle>
              <line x1="12" y1="16" x2="12" y2="12"></line>
              <line x1="12" y1="8" x2="12.01" y2="8"></line>
            </svg>
            <p>This panel can display additional context, character info, conversation history, or any other relevant information.</p>
          </div>
        </div>
      </div>

      <!-- Settings Drawer Toggle Button (Right) -->
      <button class="drawer-toggle" id="drawer-toggle">
        <img class="left-arrow" src="assets/arrow-left.png" alt="arrow-compact" />
      </button>

      <!-- Settings Drawer (Right Side) -->
      <div class="settings-drawer" id="settings-drawer">
        <h2 style="font-size: 1.5rem; margin-bottom: 1.5rem; color: var(--text); font-weight: 600;">Model Settings</h2>

        <!-- Model Selection -->
        <div class="setting-group">
          <label class="setting-label">Model</label>
          <div class="model-dropdown" id="model-dropdown">
            <button class="model-dropdown-trigger" id="model-dropdown-trigger">
              <span id="selected-model-text">Loading models...</span>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M6 9l6 6 6-6"/>
              </svg>
            </button>
            <div class="model-dropdown-menu" id="model-dropdown-menu">
              <!-- Models will be populated here -->
            </div>
          </div>
        </div>

        <!-- Temperature -->
        <div class="setting-group">
          <div class="setting-header">
            <label class="setting-label">Temperature</label>
            <span class="setting-value" id="temperature-value">1.0</span>
          </div>
          <input type="range" class="setting-slider" id="temperature-slider" min="0" max="2" step="0.01" value="1.0">
          <div class="slider-labels">
            <span>0</span>
            <span>2</span>
          </div>
        </div>

        <!-- Top P -->
        <div class="setting-group">
          <div class="setting-header">
            <label class="setting-label">Top P</label>
            <span class="setting-value" id="top-p-value">1.0</span>
          </div>
          <input type="range" class="setting-slider" id="top-p-slider" min="0" max="1" step="0.01" value="1.0">
          <div class="slider-labels">
            <span>0</span>
            <span>1</span>
          </div>
        </div>

        <!-- Min P -->
        <div class="setting-group">
          <div class="setting-header">
            <label class="setting-label">Min P</label>
            <span class="setting-value" id="min-p-value">0.0</span>
          </div>
          <input type="range" class="setting-slider" id="min-p-slider" min="0" max="1" step="0.01" value="0.0">
          <div class="slider-labels">
            <span>0</span>
            <span>1</span>
          </div>
        </div>

        <!-- Top K -->
        <div class="setting-group">
          <div class="setting-header">
            <label class="setting-label">Top K</label>
            <span class="setting-value" id="top-k-value">0</span>
          </div>
          <input type="range" class="setting-slider" id="top-k-slider" min="0" max="100" step="1" value="0">
          <div class="slider-labels">
            <span>0</span>
            <span>100</span>
          </div>
        </div>

        <!-- Frequency Penalty -->
        <div class="setting-group">
          <div class="setting-header">
            <label class="setting-label">Frequency Penalty</label>
            <span class="setting-value" id="frequency-penalty-value">0.0</span>
          </div>
          <input type="range" class="setting-slider" id="frequency-penalty-slider" min="-2" max="2" step="0.1" value="0.0">
          <div class="slider-labels">
            <span>-2</span>
            <span>2</span>
          </div>
        </div>

        <!-- Presence Penalty -->
        <div class="setting-group">
          <div class="setting-header">
            <label class="setting-label">Presence Penalty</label>
            <span class="setting-value" id="presence-penalty-value">0.0</span>
          </div>
          <input type="range" class="setting-slider" id="presence-penalty-slider" min="-2" max="2" step="0.1" value="0.0">
          <div class="slider-labels">
            <span>-2</span>
            <span>2</span>
          </div>
        </div>

        <!-- Repetition Penalty -->
        <div class="setting-group">
          <div class="setting-header">
            <label class="setting-label">Repetition Penalty</label>
            <span class="setting-value" id="repetition-penalty-value">1.0</span>
          </div>
          <input type="range" class="setting-slider" id="repetition-penalty-slider" min="0" max="2" step="0.01" value="1.0">
          <div class="slider-labels">
            <span>0</span>
            <span>2</span>
          </div>
        </div>
      </div>
    `,

    models: `
      <div class="page-content" style="display: block;">
        <h1>Models</h1>
        <p>Configure and manage your AI models here.</p>
      </div>
    `,

    chats: `
      <div class="page-content" style="display: block;">
        <h1>Chats</h1>
        <p>View and manage your chat history.</p>
      </div>
    `,

    characters: `
      <div class="characters-page">
        <!-- Character List Column -->
        <div class="character-list-column">
          <!-- Character Search -->
          <div class="character-search">
            <svg class="character-search-icon" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            <input type="text" id="character-search-input" placeholder="Search characters..." />
          </div>

          <!-- Character List -->
          <div class="character-list-container">
            <div class="character-list" id="character-list">
              <!-- Characters will be dynamically added here -->
            </div>
          </div>

          <!-- Add Character Button -->
          <button class="add-character-btn" id="add-character-btn">
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
              <path stroke-linecap="round" stroke-linejoin="round" d="M12 4v16m8-8H4" />
            </svg>
            Add Character
          </button>
        </div>

        <!-- Character Card Column -->
        <div class="character-card-column">
          <!-- Welcome Message -->
          <div class="character-welcome" id="character-welcome">
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
            </svg>
            <h2>Welcome to Characters</h2>
            <p>Create and manage AI characters for your conversations. Click "Add Character" to get started, or select an existing character from the list.</p>
          </div>

          <!-- Character Card (Embedded) -->
          <div class="character-card" id="character-card">
          <!-- Card Header -->
          <div class="card-header">
            <div class="header-left">
              <div class="avatar-container">
                <div class="avatar" id="header-avatar">
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                  </svg>
                </div>
                <div class="avatar-edit-btn" id="avatar-edit-btn">
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4" />
                  </svg>
                </div>
              </div>
              <h2 class="character-name" id="character-name-display">Character name</h2>
            </div>
            <button class="close-btn" id="character-card-close-btn">
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          <!-- Tabs Header -->
          <div class="tabs-header">
            <button class="character-tab-button active" data-tab="profile">Profile</button>
            <button class="character-tab-button" data-tab="background">Background</button>
            <button class="character-tab-button" data-tab="chats">Chats</button>
            <button class="character-tab-button" data-tab="groups">Groups</button>
            <button class="character-tab-button" data-tab="memory">Memory</button>
          </div>

          <!-- Card Body -->
          <div class="card-body">
            <!-- Left Side - Image Upload (Profile tab only) -->
            <div class="image-section" id="image-section">
              <div class="image-upload-area" id="image-upload-area">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                </svg>
                <span>Click to upload image</span>
              </div>
              <input type="file" id="character-image-input" accept="image/*" style="display: none;">
            </div>

            <!-- Right Side - Form Content -->
            <div class="content-section" id="content-section">
              <div class="tab-content">
                <!-- Profile Tab -->
                <div class="tab-panel active" id="profile-panel">
                  <div class="form-group">
                    <label class="form-label">Global Roleplay System Prompt</label>
                    <textarea class="form-textarea" id="character-global-prompt" placeholder="Enter global roleplay system prompt">You are {character.name}, a roleplay actor engaging in a conversation with {user.name}. Your replies should be written in a conversational format, taking on the personality and characteristics of {character.name}.</textarea>
                  </div>

                  <div class="form-group">
                    <label class="form-label">Character Name</label>
                    <input type="text" class="form-input" id="character-name-input" placeholder="Enter character name" value="Character name">
                  </div>

                  <div class="form-group">
                    <label class="form-label">Voice</label>
                    <select class="form-select" id="character-voice">
                      <option value="">Select voice</option>
                      <option value="voice1">Voice 1</option>
                      <option value="voice2">Voice 2</option>
                      <option value="voice3">Voice 3</option>
                    </select>
                  </div>

                  <div class="form-group">
                    <label class="form-label">System Prompt</label>
                    <textarea class="form-textarea" id="character-system-prompt" placeholder="Enter system prompt"></textarea>
                  </div>
                </div>

                <!-- Background Tab -->
                <div class="tab-panel" id="background-panel">
                  <div class="placeholder-content">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
                    </svg>
                    <h3>Background</h3>
                    <p>Coming soon</p>
                  </div>
                </div>

                <!-- Chats Tab -->
                <div class="tab-panel" id="chats-panel">
                  <div class="placeholder-content">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M20.25 8.511c.884.284 1.5 1.128 1.5 2.097v4.286c0 1.136-.847 2.1-1.98 2.193-.34.027-.68.052-1.02.072v3.091l-3-3c-1.354 0-2.694-.055-4.02-.163a2.115 2.115 0 01-.825-.242m9.345-8.334a2.126 2.126 0 00-.476-.095 48.64 48.64 0 00-8.048 0c-1.131.094-1.976 1.057-1.976 2.192v4.286c0 .837.46 1.58 1.155 1.951m9.345-8.334V6.637c0-1.621-1.152-3.026-2.76-3.235A48.455 48.455 0 0011.25 3c-2.115 0-4.198.137-6.24.402-1.608.209-2.76 1.614-2.76 3.235v6.226c0 1.621 1.152 3.026 2.76 3.235.577.075 1.157.14 1.74.194V21l4.155-4.155" />
                    </svg>
                    <h3>Chats</h3>
                    <p>Coming soon</p>
                  </div>
                </div>

                <!-- Groups Tab -->
                <div class="tab-panel" id="groups-panel">
                  <div class="placeholder-content">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M18 18.72a9.094 9.094 0 003.741-.479 3 3 0 00-4.682-2.72m.94 3.198l.001.031c0 .225-.012.447-.037.666A11.944 11.944 0 0112 21c-2.17 0-4.207-.576-5.963-1.584A6.062 6.062 0 016 18.719m12 0a5.971 5.971 0 00-.941-3.197m0 0A5.995 5.995 0 0012 12.75a5.995 5.995 0 00-5.058 2.772m0 0a3 3 0 00-4.681 2.72 8.986 8.986 0 003.74.477m.94-3.197a5.971 5.971 0 00-.94 3.197M15 6.75a3 3 0 11-6 0 3 3 0 016 0zm6 3a2.25 2.25 0 11-4.5 0 2.25 2.25 0 014.5 0zm-13.5 0a2.25 2.25 0 11-4.5 0 2.25 2.25 0 014.5 0z" />
                    </svg>
                    <h3>Groups</h3>
                    <p>Coming soon</p>
                  </div>
                </div>

                <!-- Memory Tab -->
                <div class="tab-panel" id="memory-panel">
                  <div class="placeholder-content">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 00-2.456 2.456zM16.894 20.567L16.5 21.75l-.394-1.183a2.25 2.25 0 00-1.423-1.423L13.5 18.75l1.183-.394a2.25 2.25 0 001.423-1.423l.394-1.183.394 1.183a2.25 2.25 0 001.423 1.423l1.183.394-1.183.394a2.25 2.25 0 00-1.423 1.423z" />
                    </svg>
                    <h3>Memory</h3>
                    <p>Coming soon</p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- Card Footer -->
          <div class="card-footer">
            <div class="footer-left">
              <button class="btn btn-danger" id="delete-character-btn">Delete</button>
            </div>
            <div class="footer-right">
              <button class="btn btn-secondary" id="chat-character-btn">Chat</button>
              <button class="btn btn-primary" id="save-character-btn">Save Character</button>
            </div>
          </div>
          </div>
        </div>
      </div>
    `,

    agents: `
      <div class="page-content" style="display: block;">
        <h1>Agents</h1>
        <p>Configure autonomous AI agents.</p>
      </div>
    `,

    speech: `
      <div class="speech-page">
        <!-- Left Column: Content -->
        <div class="speech-content-column">
             <div class="speech-list-placeholder">
                <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="22"/><line x1="8" y1="22" x2="16" y2="22"/></svg>
                <h3>Voice List</h3>
                <p>Coming soon: Manage your cached and generated voices here.</p>
             </div>
        </div>

        <!-- Right Column: Voice Creator -->
        <div class="speech-creator-column">
          <div class="voice-generator-card">
            <div class="voice-generator-header">
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
              </svg>
              <h2>Create Voice</h2>
            </div>
            
            <div class="voice-generator-body">
                <div class="speech-form-group">
                    <label class="speech-form-label">Voice Name</label>
                    <input type="text" class="speech-form-input" id="speech-voice-name" placeholder="E.g., Mysterious Wizard">
                </div>

                <div class="speech-form-group">
                    <label class="speech-form-label">Method</label>
                    <div class="method-selector">
                        <label class="method-card">
                            <input type="radio" name="speech-method" value="clone" id="speech-method-clone" checked>
                            <div class="method-content">
                                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                                </svg>
                                <span>Clone</span>
                            </div>
                        </label>
                        <label class="method-card">
                            <input type="radio" name="speech-method" value="profile" id="speech-method-profile">
                            <div class="method-content">
                                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                                </svg>
                                <span>Profile</span>
                            </div>
                        </label>
                    </div>
                </div>

                <div class="speech-form-group">
                    <label class="speech-form-label">Speaker Description</label>
                    <textarea class="speech-form-textarea disabled" id="speech-speaker-description" placeholder="Not used in Clone mode" disabled></textarea>
                </div>

                <div class="speech-form-group">
                    <label class="speech-form-label">Audio Path (Clone)</label>
                    <input type="text" class="speech-form-input" id="speech-audio-path" placeholder="Path to audio file (e.g. D:\Voices\sample.wav)">
                </div>

                <div class="speech-form-group">
                    <label class="speech-form-label">Text Path (Clone)</label>
                    <input type="text" class="speech-form-input" id="speech-text-path" placeholder="Path to transcript file">
                </div>
                
                <div class="speech-form-group">
                    <label class="speech-form-label">Scene Prompt (Optional)</label>
                    <textarea class="speech-form-textarea" id="speech-scene-prompt" placeholder="Optional context for the voice generation..."></textarea>
                </div>

                <button class="create-voice-btn" id="speech-create-btn">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5v14M5 12h14"/></svg>
                    Create Voice
                </button>
            </div>
          </div>
        </div>
      </div>
    `,

    settings: `
      <div class="page-content" style="display: block;">
        <h1>Settings</h1>
        <p>Configure your application preferences.</p>
      </div>
    `,
  };

  // Get the content for the requested page
  const content = pageContent[page] || `
    <div class="page-content" style="display: block;">
      <h1>404 - Page Not Found</h1>
      <p>The page "${page}" could not be found.</p>
    </div>
  `;

  // Add fade effect
  container.style.opacity = '0';

  setTimeout(() => {
    container.innerHTML = content;
    container.style.opacity = '1';

    // Initialize editor if on home page
    if (page === 'home') {
      // Wait a bit for DOM to be ready
      setTimeout(async () => {
        initEditor();
        initDrawer();
        initInfoDrawer();

        // Initialize chat system
        await initChatSystem();
      }, 100);
    }

    // Initialize characters page
    if (page === 'characters') {
      setTimeout(() => {
        initCharacters();
      }, 100);
    }

    // Initialize speech page
    if (page === 'speech') {
      setTimeout(async () => {
        const { initSpeech } = await import('./speech.js');
        initSpeech();
      }, 100);
    }
  }, 150);
}

/**
 * Initialize settings drawer functionality (right side)
 */
function initDrawer() {
  const drawerToggle = document.getElementById('drawer-toggle');
  const drawer = document.getElementById('settings-drawer');

  if (!drawerToggle || !drawer) {
    return;
  }

  // Toggle drawer on button click
  drawerToggle.addEventListener('click', () => {
    const isOpen = drawer.classList.contains('open');

    if (isOpen) {
      drawer.classList.remove('open');
      drawerToggle.classList.remove('active');
    } else {
      drawer.classList.add('open');
      drawerToggle.classList.add('active');
    }
  });

  // Initialize model settings
  initModelSettings();
}

/**
 * Initialize info drawer functionality (left side)
 */
function initInfoDrawer() {
  const drawerToggle = document.getElementById('info-drawer-toggle');
  const drawer = document.getElementById('info-drawer');

  if (!drawerToggle || !drawer) {
    return;
  }

  // Load saved state from localStorage
  const drawerState = localStorage.getItem('infoDrawerOpen');
  if (drawerState === 'true') {
    drawer.classList.add('open');
    drawerToggle.classList.add('active');
  }

  // Toggle drawer on button click
  drawerToggle.addEventListener('click', () => {
    const isOpen = drawer.classList.contains('open');

    if (isOpen) {
      drawer.classList.remove('open');
      drawerToggle.classList.remove('active');
      localStorage.setItem('infoDrawerOpen', 'false');
    } else {
      drawer.classList.add('open');
      drawerToggle.classList.add('active');
      localStorage.setItem('infoDrawerOpen', 'true');
    }
  });
}

/**
 * Format model ID for display
 * @param {string} modelId - Model ID (e.g., "openai/gpt-4")
 * @returns {string} Formatted model name (e.g., "OpenAI / GPT-4")
 */
function formatModelName(modelId) {
  const parts = modelId.split('/');
  return parts.map(part => {
    // Capitalize first letter of each word
    return part.split('-').map(word =>
      word.charAt(0).toUpperCase() + word.slice(1)
    ).join(' ');
  }).join(' / ');
}

/**
 * Fetch OpenRouter models
 * @returns {Promise<Array>} Array of model objects
 */
async function fetchOpenRouterModels() {
  try {
    // Note: In production, you should use a proper API key from environment or settings
    const response = await fetch('https://openrouter.ai/api/v1/models', {
      method: 'GET',
      headers: {
        'Authorization': 'Bearer YOUR_API_KEY_HERE',
      },
    });

    const data = await response.json();

    // Extract and sort models by ID
    const models = data.data || [];
    return models.sort((a, b) => {
      const nameA = formatModelName(a.id).toLowerCase();
      const nameB = formatModelName(b.id).toLowerCase();
      return nameA.localeCompare(nameB);
    });
  } catch (error) {
    console.error('Error fetching OpenRouter models:', error);
    return [];
  }
}

/**
 * Initialize model settings (dropdown and sliders)
 */
async function initModelSettings() {
  // Load saved settings
  loadSettings();

  // Initialize model dropdown
  await initModelDropdown();

  // Initialize sliders
  initSliders();

  // Close dropdown when clicking outside
  document.addEventListener('click', (e) => {
    const dropdown = document.getElementById('model-dropdown');
    if (dropdown && !dropdown.contains(e.target)) {
      dropdown.classList.remove('active');
    }
  });
}

/**
 * Initialize model dropdown
 */
async function initModelDropdown() {
  const dropdownTrigger = document.getElementById('model-dropdown-trigger');
  const dropdownMenu = document.getElementById('model-dropdown-menu');
  const dropdown = document.getElementById('model-dropdown');
  const selectedModelText = document.getElementById('selected-model-text');

  if (!dropdownTrigger || !dropdownMenu || !dropdown) {
    return;
  }

  // Toggle dropdown
  dropdownTrigger.addEventListener('click', (e) => {
    e.stopPropagation();
    dropdown.classList.toggle('active');
  });

  // Fetch and populate models
  selectedModelText.textContent = 'Loading models...';
  const models = await fetchOpenRouterModels();

  if (models.length === 0) {
    selectedModelText.textContent = 'Failed to load models';
    dropdownMenu.innerHTML = '<div class="model-dropdown-item" style="color: var(--muted); cursor: default;">Failed to load models</div>';
    return;
  }

  // Populate dropdown
  dropdownMenu.innerHTML = models.map(model => `
    <div class="model-dropdown-item" data-model-id="${model.id}">
      ${formatModelName(model.id)}
    </div>
  `).join('');

  // Set initial selection
  const savedModel = localStorage.getItem('selectedModel') || models[0].id;
  selectedModelText.textContent = formatModelName(savedModel);

  // Add click handlers to model items
  const modelItems = dropdownMenu.querySelectorAll('.model-dropdown-item');
  modelItems.forEach(item => {
    item.addEventListener('click', (e) => {
      const modelId = item.getAttribute('data-model-id');
      selectedModelText.textContent = formatModelName(modelId);
      localStorage.setItem('selectedModel', modelId);

      // Remove active class from all items
      modelItems.forEach(i => i.classList.remove('active'));
      item.classList.add('active');

      dropdown.classList.remove('active');
    });

    // Highlight saved model
    if (item.getAttribute('data-model-id') === savedModel) {
      item.classList.add('active');
    }
  });
}

/**
 * Initialize all sliders
 */
function initSliders() {
  const sliders = [
    { id: 'temperature', default: 1.0 },
    { id: 'top-p', default: 1.0 },
    { id: 'min-p', default: 0.0 },
    { id: 'top-k', default: 0 },
    { id: 'frequency-penalty', default: 0.0 },
    { id: 'presence-penalty', default: 0.0 },
    { id: 'repetition-penalty', default: 1.0 },
  ];

  sliders.forEach(({ id, default: defaultValue }) => {
    const slider = document.getElementById(`${id}-slider`);
    const valueDisplay = document.getElementById(`${id}-value`);

    if (!slider || !valueDisplay) return;

    // Update display when slider changes
    slider.addEventListener('input', (e) => {
      const value = parseFloat(e.target.value);
      valueDisplay.textContent = id === 'top-k' ? value.toString() : value.toFixed(2);
      saveSettings();
    });
  });
}

/**
 * Load settings from localStorage
 */
function loadSettings() {
  const sliders = [
    'temperature',
    'top-p',
    'min-p',
    'top-k',
    'frequency-penalty',
    'presence-penalty',
    'repetition-penalty',
  ];

  sliders.forEach(id => {
    const savedValue = localStorage.getItem(id);
    if (savedValue !== null) {
      const slider = document.getElementById(`${id}-slider`);
      const valueDisplay = document.getElementById(`${id}-value`);

      if (slider && valueDisplay) {
        slider.value = savedValue;
        const value = parseFloat(savedValue);
        valueDisplay.textContent = id === 'top-k' ? value.toString() : value.toFixed(2);
      }
    }
  });
}

/**
 * Save settings to localStorage and sync to server
 */
function saveSettings() {
  const sliders = [
    'temperature',
    'top-p',
    'min-p',
    'top-k',
    'frequency-penalty',
    'presence-penalty',
    'repetition-penalty',
  ];

  sliders.forEach(id => {
    const slider = document.getElementById(`${id}-slider`);
    if (slider) {
      localStorage.setItem(id, slider.value);
    }
  });

  // Sync to server
  syncModelSettings();
}

// ============================================
// CHAT SYSTEM INITIALIZATION
// ============================================

/**
 * Initialize the chat system (WebSocket, TTS, Chat UI)
 */
async function initChatSystem() {
  console.log('[Main] Initializing chat system...');

  try {
    // 1. Initialize TTS playback
    await ttsAudio.initTTSPlayback();
    console.log('[Main] TTS playback initialized');

    // 2. Connect to WebSocket
    await websocket.connect();
    console.log('[Main] WebSocket connected');

    // 3. Initialize chat UI
    chat.initChat();
    console.log('[Main] Chat UI initialized');

    // 4. Sync model settings on connection
    websocket.onConnectionChange((status) => {
      if (status === 'connected') {
        syncModelSettings();
        websocket.refreshActiveCharacters();
      }
    });

    // 5. Initial sync if already connected
    if (websocket.isConnected()) {
      syncModelSettings();
      websocket.refreshActiveCharacters();
    }

    console.log('[Main] Chat system ready');

  } catch (error) {
    console.error('[Main] Failed to initialize chat system:', error);
  }
}

/**
 * Sync model settings to server via WebSocket
 */
function syncModelSettings() {
  if (!websocket.isConnected()) {
    return;
  }

  const settings = {
    model: localStorage.getItem('selectedModel') || 'meta-llama/llama-3.1-8b-instruct',
    temperature: parseFloat(localStorage.getItem('temperature') || '1.0'),
    top_p: parseFloat(localStorage.getItem('top-p') || '1.0'),
    min_p: parseFloat(localStorage.getItem('min-p') || '0.0'),
    top_k: parseInt(localStorage.getItem('top-k') || '0'),
    frequency_penalty: parseFloat(localStorage.getItem('frequency-penalty') || '0.0'),
    presence_penalty: parseFloat(localStorage.getItem('presence-penalty') || '0.0'),
    repetition_penalty: parseFloat(localStorage.getItem('repetition-penalty') || '1.0'),
  };

  websocket.updateModelSettings(settings);
  console.log('[Main] Model settings synced:', settings.model);
}
