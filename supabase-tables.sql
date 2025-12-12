SUPABASE_URL = 'https://jslevsbvapopncjehhva.supabase.co';
SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImpzbGV2c2J2YXBvcG5jamVoaHZhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTgwNTQwOTMsImV4cCI6MjA3MzYzMDA5M30.DotbJM3IrvdVzwfScxOtsSpxq0xsj7XxI3DvdiqDSrE';

create table public.characters (
  name text not null,
  id text not null,
  voice text null,
  system_prompt text null,
  is_active boolean null default true,
  image_url text null,
  images jsonb null,
  created_at timestamp with time zone null default now(),
  updated_at timestamp with time zone null default now(),
  last_message text null,
  constraint characters_pkey primary key (id),
  constraint characters_character_id_key unique (id)
) TABLESPACE pg_default;

create index IF not exists idx_characters_character_id on public.characters using btree (id) TABLESPACE pg_default;

create trigger update_characters_updated_at BEFORE
update on characters for EACH row
execute FUNCTION update_updated_at_column ();

create table public.conversations (
  conversation_id uuid not null default extensions.uuid_generate_v4 (),
  title text null,
  active_characters jsonb null default '[]'::jsonb,
  created_at timestamp with time zone null default now(),
  updated_at timestamp with time zone null default now(),
  constraint conversations_pkey primary key (conversation_id)
) TABLESPACE pg_default;

create trigger update_conversations_updated_at BEFORE
update on conversations for EACH row
execute FUNCTION update_updated_at_column ();

create table public.messages (
  message_id uuid not null default extensions.uuid_generate_v4 (),
  conversation_id uuid not null,
  role text not null,
  name text null,
  content text not null,
  created_at timestamp with time zone null default now(),
  updated_at timestamp with time zone null default now(),
  character_id text null,
  constraint messages_pkey primary key (message_id),
  constraint messages_character_id_fkey foreign KEY (character_id) references characters (id),
  constraint messages_conversation_id_fkey foreign KEY (conversation_id) references conversations (conversation_id) on delete CASCADE
) TABLESPACE pg_default;

create index IF not exists idx_messages_conversation_id on public.messages using btree (conversation_id) TABLESPACE pg_default;

create index IF not exists idx_messages_character_name on public.messages using btree (name) TABLESPACE pg_default;

create index IF not exists idx_messages_created_at on public.messages using btree (created_at) TABLESPACE pg_default;

create trigger update_messages_updated_at BEFORE
update on messages for EACH row
execute FUNCTION update_updated_at_column ();

create table public.voices (
  voice text not null,
  method text null,
  audio_path text null,
  text_path text null,
  speaker_desc text null default ''::text,
  scene_prompt text null default ''::text,
  audio_tokens jsonb null,
  created_at timestamp with time zone null default now(),
  updated_at timestamp with time zone null default now(),
  constraint voices_pkey primary key (voice)
) TABLESPACE pg_default;

create trigger update_voices_updated_at BEFORE
update on voices for EACH row
execute FUNCTION update_updated_at_column ();

create table public.conversations (
  conversation_id uuid not null default extensions.uuid_generate_v4 (),
  created_at timestamp with time zone null default now(),
  updated_at timestamp with time zone null default now(),
  title text null,
  active_characters jsonb[] null,
  constraint conversations_pkey primary key (conversation_id)
) TABLESPACE pg_default;

create trigger update_conversations_updated_at BEFORE
update on conversations for EACH row
execute FUNCTION update_updated_at_column ();
