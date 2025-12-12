/**
 * Character Cache System
 * Provides instant access to characters and voices with automatic sync
 */

import { supabase, TABLES, handleSupabaseError, logError } from './supabase.js';

class CharacterCache {
  constructor() {
    // In-memory storage
    this.characters = new Map();  // id → character object
    this.voices = new Map();      // voice name → voice object

    // State tracking
    this.isInitialized = false;
    this.lastSync = null;
    this.isLoading = false;

    // Event handlers for cache updates
    this.eventHandlers = new Map();
  }

  /**
   * Initialize cache by loading all data from Supabase
   * @returns {Promise<{characters: Array, voices: Array}>}
   */
  async initialize() {
    if (this.isInitialized) {
      console.log('Cache already initialized');
      return this.getAll();
    }

    this.isLoading = true;
    console.log('Initializing character cache...');

    try {
      // Load characters and voices in parallel
      const [charactersResult, voicesResult] = await Promise.all([
        supabase.from(TABLES.CHARACTERS).select('*').order('created_at', { ascending: false }),
        supabase.from(TABLES.VOICES).select('*').order('created_at', { ascending: false }),
      ]);

      // Handle errors
      if (charactersResult.error) throw charactersResult.error;
      if (voicesResult.error) throw voicesResult.error;

      // Populate cache
      const characters = charactersResult.data || [];
      const voices = voicesResult.data || [];

      characters.forEach(char => this.characters.set(char.id, char));
      voices.forEach(voice => this.voices.set(voice.voice, voice));

      this.isInitialized = true;
      this.lastSync = Date.now();
      this.isLoading = false;

      console.log(`✅ Cache initialized: ${characters.length} characters, ${voices.length} voices`);

      // Emit initialization event
      this.emit('cache:initialized', { characters, voices });

      return this.getAll();

    } catch (error) {
      this.isLoading = false;
      logError('CharacterCache.initialize', error);
      throw error;
    }
  }

  /**
   * Get all cached data
   * @returns {{characters: Array, voices: Array}}
   */
  getAll() {
    return {
      characters: Array.from(this.characters.values()),
      voices: Array.from(this.voices.values()),
    };
  }

  /**
   * Get a specific character by ID (instant, from cache)
   * @param {string} id - Character ID
   * @returns {Object|null}
   */
  getCharacter(id) {
    return this.characters.get(id) || null;
  }

  /**
   * Get all characters (instant, from cache)
   * @returns {Array}
   */
  getAllCharacters() {
    return Array.from(this.characters.values());
  }

  /**
   * Get a specific voice by name (instant, from cache)
   * @param {string} voiceName - Voice name
   * @returns {Object|null}
   */
  getVoice(voiceName) {
    return this.voices.get(voiceName) || null;
  }

  /**
   * Get all voices (instant, from cache)
   * @returns {Array}
   */
  getAllVoices() {
    return Array.from(this.voices.values());
  }

  /**
   * Generate a unique character ID based on name
   * Format: lowercase-name-###
   * Example: "Jenna Haze" -> "jenna-haze-001"
   * @param {string} name - Character name
   * @returns {string} Generated ID
   */
  generateCharacterId(name) {
    // Convert to lowercase, replace spaces/special chars with dashes
    const baseName = name
      .toLowerCase()
      .trim()
      .replace(/[^a-z0-9\s-]/g, '')
      .replace(/\s+/g, '-')
      .replace(/-+/g, '-')
      .replace(/^-|-$/g, '');

    // Find existing characters with the same base name
    const existingIds = Array.from(this.characters.keys())
      .filter(id => id.startsWith(baseName + '-'))
      .sort();

    // If no existing characters with this base name, use -001
    if (existingIds.length === 0) {
      return `${baseName}-001`;
    }

    // Extract numbers from existing IDs and find the highest
    const numbers = existingIds.map(id => {
      const match = id.match(/-(\d{3})$/);
      return match ? parseInt(match[1], 10) : 0;
    });

    const maxNumber = Math.max(...numbers);
    const nextNumber = (maxNumber + 1).toString().padStart(3, '0');

    return `${baseName}-${nextNumber}`;
  }

  /**
   * Create a new character (optimistic update + background sync)
   * @param {Object} characterData - Character data (without ID)
   * @returns {Promise<Object>} Created character with generated ID
   */
  async createCharacter(characterData) {
    console.log('Creating character:', characterData.name);

    try {
      // Generate ID based on character name
      const id = this.generateCharacterId(characterData.name);

      // Add ID to character data
      const dataWithId = {
        ...characterData,
        id: id
      };

      console.log('Generated ID:', id);

      // Create in database
      const { data, error } = await supabase
        .from(TABLES.CHARACTERS)
        .insert([dataWithId])
        .select()
        .single();

      if (error) throw error;

      // Add to cache
      this.characters.set(data.id, data);

      console.log('✅ Character created:', data.id);

      // Emit event
      this.emit('character:created', data);

      return data;

    } catch (error) {
      logError('CharacterCache.createCharacter', error);
      throw error;
    }
  }

  /**
   * Update an existing character (optimistic update + background sync)
   * @param {string} id - Character ID
   * @param {Object} updates - Fields to update
   * @returns {Promise<Object>} Updated character
   */
  async updateCharacter(id, updates) {
    console.log('Updating character:', id);

    // Store original for rollback
    const original = this.characters.get(id);
    if (!original) {
      throw new Error(`Character not found: ${id}`);
    }

    // Optimistic update (update cache immediately)
    const optimistic = { ...original, ...updates };
    this.characters.set(id, optimistic);

    // Emit optimistic update event
    this.emit('character:updated', optimistic);

    try {
      // Sync to database in background
      const { data, error } = await supabase
        .from(TABLES.CHARACTERS)
        .update(updates)
        .eq('id', id)
        .select()
        .single();

      if (error) throw error;

      // Update cache with server response (has timestamps, etc.)
      this.characters.set(data.id, data);

      console.log('✅ Character updated:', data.id);

      // Emit confirmed update event
      this.emit('character:updated:confirmed', data);

      return data;

    } catch (error) {
      // Rollback on error
      console.error('Update failed, rolling back:', error);
      this.characters.set(id, original);
      this.emit('character:updated', original);

      logError('CharacterCache.updateCharacter', error);
      throw error;
    }
  }

  /**
   * Delete a character (optimistic update + background sync)
   * @param {string} id - Character ID
   * @returns {Promise<void>}
   */
  async deleteCharacter(id) {
    console.log('Deleting character:', id);

    // Store original for rollback
    const original = this.characters.get(id);
    if (!original) {
      throw new Error(`Character not found: ${id}`);
    }

    // Optimistic delete (remove from cache immediately)
    this.characters.delete(id);

    // Emit optimistic delete event
    this.emit('character:deleted', { id, character: original });

    try {
      // Sync to database in background
      const { error } = await supabase
        .from(TABLES.CHARACTERS)
        .delete()
        .eq('id', id);

      if (error) throw error;

      console.log('✅ Character deleted:', id);

      // Emit confirmed delete event
      this.emit('character:deleted:confirmed', { id });

    } catch (error) {
      // Rollback on error
      console.error('Delete failed, rolling back:', error);
      this.characters.set(id, original);
      this.emit('character:created', original); // Re-add to UI

      logError('CharacterCache.deleteCharacter', error);
      throw error;
    }
  }

  /**
   * Create a new voice (optimistic update + background sync)
   * @param {Object} voiceData - Voice data
   * @returns {Promise<Object>} Created voice
   */
  async createVoice(voiceData) {
    console.log('Creating voice:', voiceData.voice);

    try {
      // Create in database
      const { data, error } = await supabase
        .from(TABLES.VOICES)
        .insert([voiceData])
        .select()
        .single();

      if (error) throw error;

      // Add to cache
      this.voices.set(data.voice, data);

      console.log('✅ Voice created:', data.voice);

      // Emit event
      this.emit('voice:created', data);

      return data;

    } catch (error) {
      logError('CharacterCache.createVoice', error);
      throw error;
    }
  }

  /**
   * Update an existing voice
   * @param {string} voiceName - Voice name
   * @param {Object} updates - Fields to update
   * @returns {Promise<Object>} Updated voice
   */
  async updateVoice(voiceName, updates) {
    console.log('Updating voice:', voiceName);

    // Store original for rollback
    const original = this.voices.get(voiceName);
    if (!original) {
      throw new Error(`Voice not found: ${voiceName}`);
    }

    // Optimistic update
    const optimistic = { ...original, ...updates };
    this.voices.set(voiceName, optimistic);
    this.emit('voice:updated', optimistic);

    try {
      // Sync to database
      const { data, error } = await supabase
        .from(TABLES.VOICES)
        .update(updates)
        .eq('voice', voiceName)
        .select()
        .single();

      if (error) throw error;

      // Update cache with server response
      this.voices.set(data.voice, data);

      console.log('✅ Voice updated:', data.voice);
      this.emit('voice:updated:confirmed', data);

      return data;

    } catch (error) {
      // Rollback
      this.voices.set(voiceName, original);
      this.emit('voice:updated', original);

      logError('CharacterCache.updateVoice', error);
      throw error;
    }
  }

  /**
   * Delete a voice
   * @param {string} voiceName - Voice name
   * @returns {Promise<void>}
   */
  async deleteVoice(voiceName) {
    console.log('Deleting voice:', voiceName);

    const original = this.voices.get(voiceName);
    if (!original) {
      throw new Error(`Voice not found: ${voiceName}`);
    }

    // Optimistic delete
    this.voices.delete(voiceName);
    this.emit('voice:deleted', { voice: voiceName, data: original });

    try {
      const { error } = await supabase
        .from(TABLES.VOICES)
        .delete()
        .eq('voice', voiceName);

      if (error) throw error;

      console.log('✅ Voice deleted:', voiceName);
      this.emit('voice:deleted:confirmed', { voice: voiceName });

    } catch (error) {
      // Rollback
      this.voices.set(voiceName, original);
      this.emit('voice:created', original);

      logError('CharacterCache.deleteVoice', error);
      throw error;
    }
  }

  /**
   * Manually update a character in cache (for real-time sync)
   * @param {Object} character - Character object
   */
  updateCharacterInCache(character) {
    this.characters.set(character.id, character);
    this.emit('character:updated:external', character);
  }

  /**
   * Manually add a character to cache (for real-time sync)
   * @param {Object} character - Character object
   */
  addCharacterToCache(character) {
    this.characters.set(character.id, character);
    this.emit('character:created:external', character);
  }

  /**
   * Manually remove a character from cache (for real-time sync)
   * @param {string} id - Character ID
   */
  removeCharacterFromCache(id) {
    const character = this.characters.get(id);
    this.characters.delete(id);
    this.emit('character:deleted:external', { id, character });
  }

  /**
   * Manually update a voice in cache (for real-time sync)
   * @param {Object} voice - Voice object
   */
  updateVoiceInCache(voice) {
    this.voices.set(voice.voice, voice);
    this.emit('voice:updated:external', voice);
  }

  /**
   * Manually add a voice to cache (for real-time sync)
   * @param {Object} voice - Voice object
   */
  addVoiceToCache(voice) {
    this.voices.set(voice.voice, voice);
    this.emit('voice:created:external', voice);
  }

  /**
   * Manually remove a voice from cache (for real-time sync)
   * @param {string} voiceName - Voice name
   */
  removeVoiceFromCache(voiceName) {
    const voice = this.voices.get(voiceName);
    this.voices.delete(voiceName);
    this.emit('voice:deleted:external', { voice: voiceName, data: voice });
  }

  /**
   * Register an event handler
   * @param {string} eventName - Event name
   * @param {Function} handler - Callback function
   */
  on(eventName, handler) {
    if (!this.eventHandlers.has(eventName)) {
      this.eventHandlers.set(eventName, []);
    }
    this.eventHandlers.get(eventName).push(handler);
  }

  /**
   * Unregister an event handler
   * @param {string} eventName - Event name
   * @param {Function} handler - Callback function
   */
  off(eventName, handler) {
    if (!this.eventHandlers.has(eventName)) return;

    const handlers = this.eventHandlers.get(eventName);
    const index = handlers.indexOf(handler);
    if (index > -1) {
      handlers.splice(index, 1);
    }
  }

  /**
   * Emit an event to all registered handlers
   * @param {string} eventName - Event name
   * @param {*} data - Event data
   */
  emit(eventName, data) {
    const handlers = this.eventHandlers.get(eventName) || [];
    handlers.forEach(handler => {
      try {
        handler(data);
      } catch (error) {
        console.error(`Error in event handler for ${eventName}:`, error);
      }
    });
  }

  /**
   * Clear the cache (useful for logout or testing)
   */
  clear() {
    this.characters.clear();
    this.voices.clear();
    this.isInitialized = false;
    this.lastSync = null;
    console.log('Cache cleared');
  }

  /**
   * Get cache statistics
   * @returns {Object} Cache stats
   */
  getStats() {
    return {
      characters: this.characters.size,
      voices: this.voices.size,
      isInitialized: this.isInitialized,
      lastSync: this.lastSync,
      memorySize: this.estimateMemorySize(),
    };
  }

  /**
   * Estimate memory usage (rough calculation)
   * @returns {string} Memory size estimate
   */
  estimateMemorySize() {
    const charBytes = JSON.stringify(Array.from(this.characters.values())).length;
    const voiceBytes = JSON.stringify(Array.from(this.voices.values())).length;
    const totalBytes = charBytes + voiceBytes;

    if (totalBytes < 1024) return `${totalBytes} B`;
    if (totalBytes < 1024 * 1024) return `${(totalBytes / 1024).toFixed(2)} KB`;
    return `${(totalBytes / (1024 * 1024)).toFixed(2)} MB`;
  }
}

// Export singleton instance
export const characterCache = new CharacterCache();

// Export class for testing
export { CharacterCache };
