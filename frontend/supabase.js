/**
 * Supabase Client Configuration
 * Single source of truth for database access
 */

import { createClient } from 'https://esm.sh/@supabase/supabase-js@2';

// Supabase Configuration
const SUPABASE_URL = 'https://jslevsbvapopncjehhva.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImpzbGV2c2J2YXBvcG5jamVoaHZhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTgwNTQwOTMsImV4cCI6MjA3MzYzMDA5M30.DotbJM3IrvdVzwfScxOtsSpxq0xsj7XxI3DvdiqDSrE';

// Initialize Supabase client
export const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

/**
 * Database table names for reference
 */
export const TABLES = {
  CHARACTERS: 'characters',
  VOICES: 'voices',
  CHATS: 'chats',
  MESSAGES: 'messages',
};

/**
 * Helper function to handle Supabase errors consistently
 * @param {Error} error - Supabase error object
 * @returns {string} User-friendly error message
 */
export function handleSupabaseError(error) {
  if (!error) return 'An unknown error occurred';

  // Handle specific error codes
  if (error.code === 'PGRST116') {
    return 'Resource not found';
  }
  if (error.code === '23505') {
    return 'Resource already exists';
  }
  if (error.code === '23503') {
    return 'Cannot delete - resource is in use';
  }

  // Return the error message or a generic message
  return error.message || 'Database operation failed';
}

/**
 * Log errors for debugging
 * @param {string} context - Context where error occurred
 * @param {Error} error - Error object
 */
export function logError(context, error) {
  console.error(`[Supabase Error - ${context}]`, {
    message: error?.message,
    code: error?.code,
    details: error?.details,
    hint: error?.hint,
  });
}
