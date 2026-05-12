create extension if not exists "pgcrypto";

create table if not exists public.notebooks (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid,
  title text not null,
  description text not null default '',
  cover_image text not null default '',
  accent_color text not null default '#5f6fff',
  pinned boolean not null default false,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  last_opened_at timestamptz not null default now()
);

create table if not exists public.notebook_sources (
  id uuid primary key default gen_random_uuid(),
  notebook_id uuid not null references public.notebooks(id) on delete cascade,
  kind text not null default 'file',
  title text not null,
  filename text not null,
  content_type text not null,
  size_bytes bigint not null default 0,
  storage_path text not null default '',
  public_url text not null default '',
  local_path text not null default '',
  created_at timestamptz not null default now()
);

create table if not exists public.notebook_messages (
  id uuid primary key default gen_random_uuid(),
  notebook_id uuid not null references public.notebooks(id) on delete cascade,
  role text not null,
  content text not null,
  kind text not null default 'chat',
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists public.notebook_runs (
  id uuid primary key default gen_random_uuid(),
  notebook_id uuid not null references public.notebooks(id) on delete cascade,
  source_id uuid references public.notebook_sources(id) on delete set null,
  run_id text not null unique,
  title text not null,
  status text not null default 'queued',
  num_questions integer not null default 5,
  manifest jsonb not null default '{}'::jsonb,
  summary jsonb not null default '{}'::jsonb,
  artifact_paths jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

alter table public.notebooks enable row level security;
alter table public.notebook_sources enable row level security;
alter table public.notebook_messages enable row level security;
alter table public.notebook_runs enable row level security;

insert into storage.buckets (id, name, public)
values ('quiz-assets', 'quiz-assets', true)
on conflict (id) do nothing;

drop policy if exists "public read quiz assets" on storage.objects;

create policy "public read quiz assets"
on storage.objects for select
to public
using (bucket_id = 'quiz-assets');

-- Per-user notebooks: store Supabase Auth user id (uuid). No FK to auth.users so PostgREST inserts stay reliable.
alter table public.notebooks add column if not exists owner_id uuid;
create index if not exists notebooks_owner_id_idx on public.notebooks (owner_id);
