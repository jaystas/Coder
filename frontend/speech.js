/**
 * speech.js - Speech Page Functionality
 * Handles voice generation and speech settings
 */

import { characterCache } from './characterCache.js';
import { handleSupabaseError } from './supabase.js';

let isSpeechInitialized = false;

/**
 * Initialize the speech page
 */
export function initSpeech() {
    if (isSpeechInitialized) return;

    console.log('Initializing speech page...');

    setupEventListeners();

    // Initialize state
    handleMethodChange();

    isSpeechInitialized = true;
}

/**
 * Setup event listeners for the speech page
 */
function setupEventListeners() {
    // Method radio buttons
    const cloneRadio = document.getElementById('speech-method-clone');
    const profileRadio = document.getElementById('speech-method-profile');

    if (cloneRadio && profileRadio) {
        cloneRadio.addEventListener('change', handleMethodChange);
        profileRadio.addEventListener('change', handleMethodChange);
    }

    // Create Voice button
    const createBtn = document.getElementById('speech-create-btn');
    if (createBtn) {
        createBtn.addEventListener('click', handleCreateVoice);
    }
}

/**
 * Handle voice creation method change
 */
function handleMethodChange() {
    const cloneRadio = document.getElementById('speech-method-clone');
    const profileRadio = document.getElementById('speech-method-profile');

    // Inputs
    const speakerDesc = document.getElementById('speech-speaker-description');
    const audioPath = document.getElementById('speech-audio-path');
    const textPath = document.getElementById('speech-text-path');

    if (!cloneRadio || !profileRadio) return;

    if (cloneRadio.checked) {
        // Clone Method Active
        if (speakerDesc) {
            speakerDesc.disabled = true;
            speakerDesc.classList.add('disabled');
            speakerDesc.placeholder = "Not used in Clone mode";
        }

        if (audioPath) {
            audioPath.disabled = false;
            audioPath.classList.remove('disabled');
        }

        if (textPath) {
            textPath.disabled = false;
            textPath.classList.remove('disabled');
        }

    } else if (profileRadio.checked) {
        // Profile/Description Method Active
        if (speakerDesc) {
            speakerDesc.disabled = false;
            speakerDesc.classList.remove('disabled');
            speakerDesc.placeholder = "Describe the speaker's voice characteristics, tone, accent, and style...";
        }

        if (audioPath) {
            audioPath.disabled = true;
            audioPath.classList.add('disabled');
        }

        if (textPath) {
            textPath.disabled = true;
            textPath.classList.add('disabled');
        }
    }
}

/**
 * Generate a voice name
 */
function generateVoiceName(nameInput) {
    const baseName = nameInput
        .toLowerCase()
        .trim()
        .replace(/[^a-z0-9\s-]/g, '')
        .replace(/\s+/g, '-')
        .replace(/-+/g, '-')
        .replace(/^-|-$/g, '');

    const timestamp = Date.now();
    return `${baseName}-voice-${timestamp}`;
}

/**
 * Handle Create Voice button click
 */
async function handleCreateVoice() {
    const nameInput = document.getElementById('speech-voice-name');
    const cloneRadio = document.getElementById('speech-method-clone');
    const speakerDesc = document.getElementById('speech-speaker-description');
    const scenePrompt = document.getElementById('speech-scene-prompt');
    const audioPath = document.getElementById('speech-audio-path');
    const textPath = document.getElementById('speech-text-path');
    const createBtn = document.getElementById('speech-create-btn');

    // Validation
    if (!nameInput || !nameInput.value.trim()) {
        showNotification('Validation Error', 'Please enter a name for this voice', 'error');
        return;
    }

    const method = cloneRadio.checked ? 'clone' : 'description';

    if (method === 'description' && (!speakerDesc || !speakerDesc.value.trim())) {
        showNotification('Validation Error', 'Speaker description is required for Profile method', 'error');
        return;
    }

    if (method === 'clone' && ((!audioPath || !audioPath.value.trim()) || (!textPath || !textPath.value.trim()))) {
        showNotification('Validation Error', 'Audio path and text path are required for Clone method', 'error');
        return;
    }

    const voiceName = generateVoiceName(nameInput.value);

    const voiceData = {
        voice: voiceName,
        method: method,
        speaker_desc: speakerDesc ? speakerDesc.value : '',
        scene_prompt: scenePrompt ? scenePrompt.value : '',
        audio_path: audioPath ? audioPath.value : '',
        text_path: textPath ? textPath.value : ''
    };

    // UI Feedback
    if (createBtn) {
        createBtn.disabled = true;
        createBtn.innerHTML = `<span class="loading-spinner-small"></span> Creating...`;
    }

    try {
        // Use characterCache to create voice (assuming it handles the backend call)
        const data = await characterCache.createVoice(voiceData);

        console.log('Voice created:', data);

        showNotification(
            'Voice Created',
            `Voice "${nameInput.value}" (${data.voice}) created successfully!`,
            'success'
        );

        // Reset form
        nameInput.value = '';
        if (speakerDesc) speakerDesc.value = '';
        if (scenePrompt) scenePrompt.value = '';
        if (audioPath) audioPath.value = '';
        if (textPath) textPath.value = '';

    } catch (error) {
        console.error('Error creating voice:', error);
        const errorMessage = handleSupabaseError(error);
        showNotification('Error Creating Voice', errorMessage, 'error');
    } finally {
        if (createBtn) {
            createBtn.disabled = false;
            createBtn.textContent = 'Create Voice';
        }
    }
}

/**
 * Show notification helper (reusing global one if available or creating simple alert)
 * Ideally this should use the same notification system as characters.js
 */
function showNotification(title, message, type = 'info') {
    // Check if there's a global notification function attached to window or exported
    // characters.js defined showNotification internally. 
    // We can define a simple one here or try to reuse if we refactored.
    // For now, I'll create a simple one here similar to characters.js

    const container = document.getElementById('notification-container');
    if (!container) return; // Should be created by initCharacters or main

    const notification = document.createElement('div');
    notification.className = `notification notification-${type} slide-in`;

    let icon = '';
    if (type === 'success') icon = '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" /></svg>';
    else if (type === 'error') icon = '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" /></svg>';
    else icon = '<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>';

    notification.innerHTML = `
        <div class="notification-icon">${icon}</div>
        <div class="notification-content">
            <div class="notification-title">${title}</div>
            <div class="notification-message">${message}</div>
        </div>
        <button class="notification-close">&times;</button>
    `;

    container.appendChild(notification);

    // Close button
    notification.querySelector('.notification-close').addEventListener('click', () => {
        notification.classList.add('slide-out');
        setTimeout(() => notification.remove(), 300);
    });

    // Auto close
    setTimeout(() => {
        if (notification.parentNode) {
            notification.classList.add('slide-out');
            setTimeout(() => notification.remove(), 300);
        }
    }, 5000);
}
