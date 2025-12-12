/**
 * Real-Time Sync System
 * Listens to Supabase real-time events and syncs with character cache
 */

import { supabase, TABLES } from './supabase.js';
import { characterCache } from './characterCache.js';

class RealtimeSync {
  constructor() {
    this.subscriptions = [];
    this.isActive = false;
    this.channels = {
      characters: null,
      voices: null,
    };
  }

  /**
   * Start real-time sync for all tables
   */
  start() {
    if (this.isActive) {
      console.log('Real-time sync already active');
      return;
    }

    console.log('Starting real-time sync...');

    this.subscribeToCharacters();
    this.subscribeToVoices();

    this.isActive = true;
    console.log('✅ Real-time sync active');
  }

  /**
   * Subscribe to character table changes
   */
  subscribeToCharacters() {
    console.log('Subscribing to characters table...');

    const channel = supabase
      .channel('characters-changes')
      .on(
        'postgres_changes',
        {
          event: '*',
          schema: 'public',
          table: TABLES.CHARACTERS,
        },
        (payload) => this.handleCharacterChange(payload)
      )
      .subscribe((status) => {
        if (status === 'SUBSCRIBED') {
          console.log('✅ Subscribed to characters table');
        } else if (status === 'CHANNEL_ERROR') {
          console.error('❌ Error subscribing to characters table');
        } else if (status === 'TIMED_OUT') {
          console.error('❌ Subscription to characters table timed out');
        }
      });

    this.channels.characters = channel;
    this.subscriptions.push(channel);
  }

  /**
   * Subscribe to voices table changes
   */
  subscribeToVoices() {
    console.log('Subscribing to voices table...');

    const channel = supabase
      .channel('voices-changes')
      .on(
        'postgres_changes',
        {
          event: '*',
          schema: 'public',
          table: TABLES.VOICES,
        },
        (payload) => this.handleVoiceChange(payload)
      )
      .subscribe((status) => {
        if (status === 'SUBSCRIBED') {
          console.log('✅ Subscribed to voices table');
        } else if (status === 'CHANNEL_ERROR') {
          console.error('❌ Error subscribing to voices table');
        } else if (status === 'TIMED_OUT') {
          console.error('❌ Subscription to voices table timed out');
        }
      });

    this.channels.voices = channel;
    this.subscriptions.push(channel);
  }

  /**
   * Handle character table changes
   * @param {Object} payload - Realtime payload from Supabase
   */
  handleCharacterChange(payload) {
    const { eventType, new: newRecord, old: oldRecord } = payload;

    console.log(`[Realtime] Character ${eventType}:`, newRecord || oldRecord);

    switch (eventType) {
      case 'INSERT':
        // Character was created (possibly in another tab/session)
        characterCache.addCharacterToCache(newRecord);
        console.log('✅ Character added to cache via realtime:', newRecord.id);
        break;

      case 'UPDATE':
        // Character was updated (possibly in another tab/session)
        characterCache.updateCharacterInCache(newRecord);
        console.log('✅ Character updated in cache via realtime:', newRecord.id);
        break;

      case 'DELETE':
        // Character was deleted (possibly in another tab/session)
        characterCache.removeCharacterFromCache(oldRecord.id);
        console.log('✅ Character removed from cache via realtime:', oldRecord.id);
        break;

      default:
        console.warn('Unknown event type:', eventType);
    }
  }

  /**
   * Handle voice table changes
   * @param {Object} payload - Realtime payload from Supabase
   */
  handleVoiceChange(payload) {
    const { eventType, new: newRecord, old: oldRecord } = payload;

    console.log(`[Realtime] Voice ${eventType}:`, newRecord || oldRecord);

    switch (eventType) {
      case 'INSERT':
        // Voice was created
        characterCache.addVoiceToCache(newRecord);
        console.log('✅ Voice added to cache via realtime:', newRecord.voice);
        break;

      case 'UPDATE':
        // Voice was updated
        characterCache.updateVoiceInCache(newRecord);
        console.log('✅ Voice updated in cache via realtime:', newRecord.voice);
        break;

      case 'DELETE':
        // Voice was deleted
        characterCache.removeVoiceFromCache(oldRecord.voice);
        console.log('✅ Voice removed from cache via realtime:', oldRecord.voice);
        break;

      default:
        console.warn('Unknown event type:', eventType);
    }
  }

  /**
   * Stop real-time sync and cleanup
   */
  stop() {
    if (!this.isActive) {
      console.log('Real-time sync not active');
      return;
    }

    console.log('Stopping real-time sync...');

    // Unsubscribe from all channels
    this.subscriptions.forEach(channel => {
      supabase.removeChannel(channel);
    });

    this.subscriptions = [];
    this.channels = {
      characters: null,
      voices: null,
    };

    this.isActive = false;
    console.log('✅ Real-time sync stopped');
  }

  /**
   * Restart real-time sync (useful for reconnection)
   */
  restart() {
    console.log('Restarting real-time sync...');
    this.stop();
    setTimeout(() => this.start(), 100);
  }

  /**
   * Check if real-time sync is active
   * @returns {boolean}
   */
  isRunning() {
    return this.isActive;
  }

  /**
   * Get subscription status
   * @returns {Object} Status of all subscriptions
   */
  getStatus() {
    return {
      isActive: this.isActive,
      subscriptions: this.subscriptions.length,
      channels: {
        characters: this.channels.characters?.state || 'inactive',
        voices: this.channels.voices?.state || 'inactive',
      },
    };
  }
}

// Export singleton instance
export const realtimeSync = new RealtimeSync();

// Export class for testing
export { RealtimeSync };
