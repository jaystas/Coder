/**
 * editor.js - Tiptap Rich Text Editor Functionality
 * Initializes and manages the rich text editor component
 */

// Import Tiptap modules from CDN
import { Editor } from 'https://esm.sh/@tiptap/core';
import StarterKit from 'https://esm.sh/@tiptap/starter-kit';
import Underline from 'https://esm.sh/@tiptap/extension-underline';
import Link from 'https://esm.sh/@tiptap/extension-link';
import TextAlign from 'https://esm.sh/@tiptap/extension-text-align';
import Highlight from 'https://esm.sh/@tiptap/extension-highlight';
import Image from 'https://esm.sh/@tiptap/extension-image';

// Import chat system modules
import * as sttAudio from './stt-audio.js';
import * as ttsAudio from './tts-audio.js';
import * as chat from './chat.js';

let editor = null;
let isMicActive = false;
let isAudioInitialized = false;

/**
 * Initialize the Tiptap editor
 */
export function initEditor() {
  const editorElement = document.querySelector('#editor');

  if (!editorElement) {
    console.warn('Editor element not found');
    return null;
  }

  // Initialize Tiptap Editor
  editor = new Editor({
    element: editorElement,
    extensions: [
      StarterKit.configure({
        heading: {
          levels: [1, 2, 3]
        }
      }),
      Underline,
      Link.configure({
        openOnClick: false,
        HTMLAttributes: {
          class: 'editor-link',
        },
      }),
      TextAlign.configure({
        types: ['heading', 'paragraph']
      }),
      Highlight.configure({
        multicolor: true
      }),
      Image
    ],
    content: '<p></p>',
    editorProps: {
      attributes: {
        class: 'prose prose-invert max-w-none'
      }
    }
  });

  // Create toolbar after editor is initialized
  createToolbar();

  // Update toolbar on editor changes
  editor.on('update', () => {
    // Optionally update toolbar state
  });

  // Close dropdowns when clicking outside
  document.addEventListener('click', () => {
    document.querySelectorAll('.dropdown-content, .color-picker').forEach(d => {
      d.classList.remove('show');
    });
  });

  return editor;
}

/**
 * Create and populate the toolbar with buttons
 */
function createToolbar() {
  const toolbar = document.getElementById('toolbar');

  if (!toolbar) {
    console.warn('Toolbar element not found');
    return;
  }

  const buttons = [
    // Undo/Redo
    {
      icon: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 7v6h6"/><path d="M21 17a9 9 0 00-9-9 9 9 0 00-6 2.3L3 13"/></svg>`,
      action: () => editor.chain().focus().undo().run(),
      isActive: () => false,
      canExecute: () => editor.can().undo()
    },
    {
      icon: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 7v6h-6"/><path d="M3 17a9 9 0 019-9 9 9 0 016 2.3l3 2.7"/></svg>`,
      action: () => editor.chain().focus().redo().run(),
      isActive: () => false,
      canExecute: () => editor.can().redo()
    },
    'divider',
    // Heading Dropdown
    {
      type: 'dropdown',
      label: 'H1',
      options: [
        { label: 'Paragraph', action: () => editor.chain().focus().setParagraph().run() },
        { label: 'Heading 1', action: () => editor.chain().focus().toggleHeading({ level: 1 }).run() },
        { label: 'Heading 2', action: () => editor.chain().focus().toggleHeading({ level: 2 }).run() },
        { label: 'Heading 3', action: () => editor.chain().focus().toggleHeading({ level: 3 }).run() }
      ]
    },
    'divider',
    // Lists Dropdown
    {
      type: 'dropdown',
      icon: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/></svg>`,
      options: [
        { label: 'Bullet List', action: () => editor.chain().focus().toggleBulletList().run() },
        { label: 'Ordered List', action: () => editor.chain().focus().toggleOrderedList().run() }
      ]
    },
    'divider',
    // Blockquote
    {
      icon: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 21c3 0 7-1 7-8V5c0-1.25-.756-2.017-2-2H4c-1.25 0-2 .75-2 1.972V11c0 1.25.75 2 2 2 1 0 1 0 1 1v1c0 1-1 2-2 2s-1 .008-1 1.031V20c0 1 0 1 1 1z"/><path d="M15 21c3 0 7-1 7-8V5c0-1.25-.757-2.017-2-2h-4c-1.25 0-2 .75-2 1.972V11c0 1.25.75 2 2 2h.75c0 2.25.25 4-2.75 4v3c0 1 0 1 1 1z"/></svg>`,
      action: () => editor.chain().focus().toggleBlockquote().run(),
      isActive: () => editor.isActive('blockquote')
    },
    'divider',
    // Bold, Italic, Strikethrough
    {
      icon: '<strong>B</strong>',
      action: () => editor.chain().focus().toggleBold().run(),
      isActive: () => editor.isActive('bold')
    },
    {
      icon: '<em>I</em>',
      action: () => editor.chain().focus().toggleItalic().run(),
      isActive: () => editor.isActive('italic')
    },
    {
      icon: '<s>S</s>',
      action: () => editor.chain().focus().toggleStrike().run(),
      isActive: () => editor.isActive('strike')
    },
    'divider',
    // Code
    {
      icon: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>`,
      action: () => editor.chain().focus().toggleCodeBlock().run(),
      isActive: () => editor.isActive('codeBlock')
    },
    // Underline
    {
      icon: '<u>U</u>',
      action: () => editor.chain().focus().toggleUnderline().run(),
      isActive: () => editor.isActive('underline')
    },
    'divider',
    // Highlight
    {
      type: 'color-picker',
      icon: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 11L12 14L22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/></svg>`,
      colors: ['#fef08a', '#fca5a5', '#a7f3d0', '#bfdbfe', '#ddd6fe']
    },
    // Link
    {
      type: 'link',
      icon: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/></svg>`,
      isActive: () => editor.isActive('link')
    },
    'divider',
    // Text Align
    {
      icon: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="17" y1="10" x2="3" y2="10"/><line x1="21" y1="6" x2="3" y2="6"/><line x1="21" y1="14" x2="3" y2="14"/><line x1="17" y1="18" x2="3" y2="18"/></svg>`,
      action: () => editor.chain().focus().setTextAlign('left').run(),
      isActive: () => editor.isActive({ textAlign: 'left' })
    },
    {
      icon: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="10" x2="6" y2="10"/><line x1="21" y1="6" x2="3" y2="6"/><line x1="21" y1="14" x2="3" y2="14"/><line x1="18" y1="18" x2="6" y2="18"/></svg>`,
      action: () => editor.chain().focus().setTextAlign('center').run(),
      isActive: () => editor.isActive({ textAlign: 'center' })
    },
    {
      icon: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="21" y1="10" x2="7" y2="10"/><line x1="21" y1="6" x2="3" y2="6"/><line x1="21" y1="14" x2="3" y2="14"/><line x1="21" y1="18" x2="7" y2="18"/></svg>`,
      action: () => editor.chain().focus().setTextAlign('right').run(),
      isActive: () => editor.isActive({ textAlign: 'right' })
    },
    'divider',
    // Add Image/Media
    {
      icon: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>`,
      label: 'Add',
      action: () => addImage()
    }
  ];

  buttons.forEach(button => {
    if (button === 'divider') {
      const divider = document.createElement('div');
      divider.className = 'toolbar-divider';
      toolbar.appendChild(divider);
    } else if (button.type === 'dropdown') {
      const dropdown = createDropdown(button);
      toolbar.appendChild(dropdown);
    } else if (button.type === 'color-picker') {
      const colorPicker = createColorPicker(button);
      toolbar.appendChild(colorPicker);
    } else if (button.type === 'link') {
      const linkButton = createLinkButton(button);
      toolbar.appendChild(linkButton);
    } else {
      const btn = createButton(button);
      toolbar.appendChild(btn);
    }
  });
}

/**
 * Create a toolbar button
 */
function createButton(config) {
  const button = document.createElement('button');
  button.className = 'toolbar-button';
  button.innerHTML = config.icon || config.label || '';

  button.addEventListener('click', () => {
    config.action();
    updateToolbar();
  });

  if (config.canExecute) {
    button.disabled = !config.canExecute();
  }

  if (config.isActive && config.isActive()) {
    button.classList.add('is-active');
  }

  return button;
}

/**
 * Create a dropdown menu
 */
function createDropdown(config) {
  const container = document.createElement('div');
  container.className = 'toolbar-dropdown';

  const button = document.createElement('button');
  button.className = 'dropdown-button';
  button.innerHTML = config.icon ? config.icon + ' <span>▼</span>' : config.label + ' <span>▼</span>';

  const content = document.createElement('div');
  content.className = 'dropdown-content';

  config.options.forEach(option => {
    const item = document.createElement('div');
    item.className = 'dropdown-item';
    item.textContent = option.label;
    item.addEventListener('click', () => {
      option.action();
      content.classList.remove('show');
      updateToolbar();
    });
    content.appendChild(item);
  });

  button.addEventListener('click', (e) => {
    e.stopPropagation();
    document.querySelectorAll('.dropdown-content').forEach(d => {
      if (d !== content) d.classList.remove('show');
    });
    content.classList.toggle('show');
  });

  container.appendChild(button);
  container.appendChild(content);
  return container;
}

/**
 * Create a color picker for text highlighting
 */
function createColorPicker(config) {
  const container = document.createElement('div');
  container.className = 'toolbar-dropdown';
  container.style.position = 'relative';

  const button = document.createElement('button');
  button.className = 'toolbar-button';
  button.innerHTML = config.icon;

  const picker = document.createElement('div');
  picker.className = 'color-picker';

  config.colors.forEach(color => {
    const colorOption = document.createElement('div');
    colorOption.className = 'color-option';
    colorOption.style.backgroundColor = color;
    colorOption.addEventListener('click', () => {
      editor.chain().focus().toggleHighlight({ color }).run();
      picker.classList.remove('show');
      updateToolbar();
    });
    picker.appendChild(colorOption);
  });

  button.addEventListener('click', (e) => {
    e.stopPropagation();
    picker.classList.toggle('show');
  });

  container.appendChild(button);
  container.appendChild(picker);
  return container;
}

/**
 * Create a link button
 */
function createLinkButton(config) {
  const button = document.createElement('button');
  button.className = 'toolbar-button';
  button.innerHTML = config.icon;

  button.addEventListener('click', () => {
    if (editor.isActive('link')) {
      editor.chain().focus().unsetLink().run();
    } else {
      const url = prompt('Enter URL:');
      if (url) {
        editor.chain().focus().setLink({ href: url }).run();
      }
    }
    updateToolbar();
  });

  if (config.isActive()) {
    button.classList.add('is-active');
  }

  return button;
}

/**
 * Update toolbar to reflect current editor state
 */
function updateToolbar() {
  // Recreate toolbar to update active states
  const toolbar = document.getElementById('toolbar');
  if (toolbar) {
    toolbar.innerHTML = '';
    createToolbar();
  }
}

/**
 * Add an image to the editor
 */
function addImage() {
  const url = prompt('Enter image URL:');
  if (url) {
    editor.chain().focus().setImage({ src: url }).run();
  }
}

/**
 * Handle microphone button click
 * Toggles between listening and idle states
 */
export async function handleMic() {
  const micButton = document.querySelector('.mic-button');

  // Initialize audio on first click (browser requirement)
  if (!isAudioInitialized) {
    console.log('[Editor] Initializing audio capture...');
    const success = await sttAudio.initAudioCapture();

    if (!success) {
      console.error('[Editor] Failed to initialize audio capture');
      return;
    }

    isAudioInitialized = true;

    // Subscribe to STT state changes for UI updates
    sttAudio.onStateChange((newStatus, oldStatus) => {
      updateMicButtonState(micButton, newStatus);
    });
  }

  if (isMicActive) {
    // Stop listening
    sttAudio.stopRecording();
    isMicActive = false;
    updateMicButtonState(micButton, 'idle');
    console.log('[Editor] Mic deactivated');
  } else {
    // Start listening
    sttAudio.startRecording();
    isMicActive = true;
    console.log('[Editor] Mic activated');
  }
}

/**
 * Update mic button visual state
 * @param {HTMLElement} button
 * @param {string} status - 'idle' | 'listening' | 'recording' | 'paused'
 */
function updateMicButtonState(button, status) {
  if (!button) return;

  // Remove all state classes
  button.classList.remove('listening', 'recording', 'paused');

  // Add appropriate class
  if (status !== 'idle') {
    button.classList.add(status);
  }
}

/**
 * Handle send button click
 * Sends text content to server via chat module
 */
export function handleSend() {
  if (!editor) {
    console.warn('Editor not initialized');
    return;
  }

  // Get plain text from editor (strip HTML tags for chat)
  const content = editor.getText().trim();

  if (!content) {
    return; // Don't send empty messages
  }

  console.log('[Editor] Sending message:', content);

  // Stop any TTS playback when user sends a message
  ttsAudio.stop();

  // Send via chat module
  chat.sendMessage(content);

  // Clear the editor after sending
  editor.commands.clearContent();
}

/**
 * Get the editor instance
 */
export function getEditor() {
  return editor;
}

/**
 * Get editor content as HTML
 */
export function getEditorContent() {
  return editor ? editor.getHTML() : '';
}

/**
 * Set editor content
 */
export function setEditorContent(content) {
  if (editor) {
    editor.commands.setContent(content);
  }
}

/**
 * Clear editor content
 */
export function clearEditorContent() {
  if (editor) {
    editor.commands.clearContent();
  }
}
