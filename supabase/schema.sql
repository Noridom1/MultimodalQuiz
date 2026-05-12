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

create table if not exists public.quiz_lists (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid,
  title text not null,
  folder_color text not null default '#f0bf57',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.quiz_list_items (
  id uuid primary key default gen_random_uuid(),
  list_id uuid not null references public.quiz_lists(id) on delete cascade,
  notebook_id uuid not null references public.notebooks(id) on delete cascade,
  run_id text not null references public.notebook_runs(run_id) on delete cascade,
  sort_order integer not null default 0,
  created_at timestamptz not null default now(),
  unique (list_id, run_id)
);

create index if not exists quiz_lists_owner_id_idx on public.quiz_lists (owner_id);
create index if not exists quiz_list_items_list_id_idx on public.quiz_list_items (list_id);

alter table public.notebooks enable row level security;
alter table public.notebook_sources enable row level security;
alter table public.notebook_messages enable row level security;
alter table public.notebook_runs enable row level security;
alter table public.quiz_lists enable row level security;
alter table public.quiz_list_items enable row level security;

drop policy if exists "quiz_lists_select_own" on public.quiz_lists;
create policy "quiz_lists_select_own"
on public.quiz_lists for select
to authenticated
using (owner_id is not null and owner_id = auth.uid());

drop policy if exists "quiz_lists_insert_own" on public.quiz_lists;
create policy "quiz_lists_insert_own"
on public.quiz_lists for insert
to authenticated
with check (owner_id is not null and owner_id = auth.uid());

drop policy if exists "quiz_lists_update_own" on public.quiz_lists;
create policy "quiz_lists_update_own"
on public.quiz_lists for update
to authenticated
using (owner_id is not null and owner_id = auth.uid())
with check (owner_id is not null and owner_id = auth.uid());

drop policy if exists "quiz_lists_delete_own" on public.quiz_lists;
create policy "quiz_lists_delete_own"
on public.quiz_lists for delete
to authenticated
using (owner_id is not null and owner_id = auth.uid());

drop policy if exists "quiz_list_items_select_via_list" on public.quiz_list_items;
create policy "quiz_list_items_select_via_list"
on public.quiz_list_items for select
to authenticated
using (
  exists (
    select 1 from public.quiz_lists ql
    where ql.id = quiz_list_items.list_id
      and ql.owner_id is not null
      and ql.owner_id = auth.uid()
  )
);

drop policy if exists "quiz_list_items_insert_via_list" on public.quiz_list_items;
create policy "quiz_list_items_insert_via_list"
on public.quiz_list_items for insert
to authenticated
with check (
  exists (
    select 1 from public.quiz_lists ql
    where ql.id = quiz_list_items.list_id
      and ql.owner_id is not null
      and ql.owner_id = auth.uid()
  )
);

drop policy if exists "quiz_list_items_delete_via_list" on public.quiz_list_items;
create policy "quiz_list_items_delete_via_list"
on public.quiz_list_items for delete
to authenticated
using (
  exists (
    select 1 from public.quiz_lists ql
    where ql.id = quiz_list_items.list_id
      and ql.owner_id is not null
      and ql.owner_id = auth.uid()
  )
);

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
