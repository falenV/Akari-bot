-- Akari schema (v3)
-- How to use: open your Supabase project -> SQL Editor -> New query, paste this whole file, and press Run. It is safe to run more than once, including over a database that already ran an earlier version of this script: it only creates or alters what's missing/changed and never deletes data
-- What it creates:
--   6 tables    long_term_memory, beliefs, belief_evidence, goals,
--               user_profiles, relationship_state
--   5 functions match_long_term_memory, find_similar_memory, reinforce_memories,
--               decay_long_term_memory, belief_evidence_health

-- Important: the bot must use the service_role key as SUPABASE_KEY. Row Level Security is enabled and public (anon) access is removed, so the anon key cannot read or write these tables. Keep the service_role key on the server only, and never commit it
-- Discord IDs (guild_id) are stored as TEXT because they are 64-bit snowflakes
-- Embeddings are 384-dimensional vectors (Xenova/all-MiniLM-L6-v2). embedding_model records which
-- model/version produced each vector, so a future embedding-model change doesn't silently mix
-- incompatible vectors in the same column -- bot.js sets it on every insert and merge.
--
-- v2 change: match_long_term_memory now also returns `status`, so the persona prompt's
-- "hedge on fading memories" instruction has something to actually check. Changing a SQL
-- function's return columns needs a DROP before the CREATE (Postgres won't let CREATE OR
-- REPLACE change output columns), so re-running this whole script on a v1 database is how you
-- pick up that change -- it's still safe to run any number of times.
--
-- v3 change (identity migration): user_profiles and relationship_state were keyed on
-- (guild_id, user_name) -- a Discord nickname, which anyone can change at any time. Changing it
-- orphaned that person's profile/rapport and silently started them over under the new name.
-- This adds a `user_id` (the stable Discord snowflake) to both tables and `subject_id` to
-- beliefs/long_term_memory. Everything about this is additive: no primary key changes, no rows
-- deleted, no old row ever becomes unreachable. bot.js resolves ids from local message history
-- and self-heals old name-only rows the moment it sees that person again; migrate-user-ids.js
-- does a one-time bulk backfill so you don't have to wait for that to happen organically.
--
-- Every function below is preceded by an explicit DROP FUNCTION IF EXISTS and defined with
-- CREATE OR REPLACE rather than a bare CREATE. Two different failure modes made this necessary,
-- both stemming from the same fact: running this script through Supabase's SQL editor does not
-- behave like one all-or-nothing transaction -- if a later statement errors, statements earlier
-- in the same run that already succeeded stay committed, they are not rolled back.
--   1. CREATE OR REPLACE can't change a function's return columns or add a parameter (Postgres
--      treats that as a different function entirely), so a signature change needs an explicit
--      DROP of the OLD signature first.
--   2. A bare CREATE (after that drop) fails if the NEW signature was already created in an
--      earlier, partially-successful run of this same script -- the DROP above only removes the
--      old signature, so if what's actually deployed is already the new one, the DROP is a
--      no-op and a bare CREATE collides with it. CREATE OR REPLACE has no such problem: it
--      succeeds whether the new signature already exists (trivially replacing itself) or not.
-- Together, drop-old-if-present + create-or-replace-target is safe to run from any starting
-- state -- fresh, an old version, a new version, or (as above) a half-applied one.




-- 0. Extension

create extension if not exists vector with schema extensions;




-- Long-term memories: what Akari has learned about people and shared moments.
create table if not exists public.long_term_memory (
    id              bigint generated always as identity primary key,
    guild_id        text             not null,
    subject         text,                                   -- usually a user name; null = general
    subject_id      text,                                   -- stable Discord id for `subject`, when resolvable; null for non-person subjects or unresolved names
    summary         text             not null,              -- one third-person sentence
    nature          text             not null default 'fact', -- fact | preference | relationship | event
    importance      double precision not null default 0.5 check (importance between 0 and 1),
    confidence      double precision not null default 0.7 check (confidence between 0 and 1),
    embedding       vector(384),
    embedding_model text,                                  -- which model produced `embedding`; set by bot.js on insert/merge
    status          text             not null default 'active'
                        check (status in ('active', 'fading', 'archived', 'forgotten')),
    evidence_count  integer          not null default 1,    -- how many times this was re-observed / merged
    access_count    integer          not null default 0,    -- how many times it was recalled
    last_accessed   timestamptz      not null default now(),
    created_at      timestamptz      not null default now()
);

-- Picks up embedding_model/subject_id on a database that already ran an earlier version of this
-- script: CREATE TABLE IF NOT EXISTS is a no-op once the table exists, so on an upgrade these
-- columns only actually appear once this ALTER runs -- which is why it has to come before the
-- indexes below that reference subject_id, not after.
alter table public.long_term_memory add column if not exists embedding_model text;
alter table public.long_term_memory add column if not exists subject_id text;

create index if not exists long_term_memory_guild_status_idx
    on public.long_term_memory (guild_id, status, created_at desc);

create index if not exists long_term_memory_subject_id_idx
    on public.long_term_memory (guild_id, subject_id) where subject_id is not null;




-- Beliefs: durable impressions about a person (scope 'user'), the server ('server'), or Akari herself ('self'). Confidence moves gradually

create table if not exists public.beliefs (
    id              bigint generated always as identity primary key,
    guild_id        text             not null,
    scope           text             not null default 'user', -- user | server | self
    subject         text,                                     -- user name for scope 'user', else null
    subject_id      text,                                     -- stable Discord id for `subject`, when resolvable
    statement       text             not null,
    confidence      double precision not null default 0.45 check (confidence between 0 and 1),
    evidence_count  integer          not null default 1,
    embedding       vector(384),
    embedding_model text,                                     -- which model produced `embedding`; set by bot.js on insert/merge
    last_updated    timestamptz      not null default now(),
    created_at      timestamptz      not null default now()
);

-- Same reasoning as long_term_memory above: this has to come before the index below that
-- references subject_id, since that column only actually exists (on an upgrade, where CREATE
-- TABLE was a no-op) once this ALTER runs.
alter table public.beliefs add column if not exists embedding_model text;
alter table public.beliefs add column if not exists subject_id text;

create index if not exists beliefs_guild_idx on public.beliefs (guild_id);

create index if not exists beliefs_subject_id_idx
    on public.beliefs (guild_id, subject_id) where subject_id is not null;


-- Links each belief to the memories that support it. Deleting a belief removes its links. memory_id deliberately has no foreign key, the model occasionally cites a memory id that does not exist, and a foreign key would make the whole batch of links fail. belief_evidence_health() ignores links to missing memories.

create table if not exists public.belief_evidence (
    belief_id  bigint not null references public.beliefs (id) on delete cascade,
    memory_id  bigint not null,
    primary key (belief_id, memory_id)
);

create index if not exists belief_evidence_memory_idx on public.belief_evidence (memory_id);


-- Goals: small things Akari is curious about or working toward 
create table if not exists public.goals (
    id            bigint generated always as identity primary key,
    guild_id      text             not null,
    goal          text             not null,
    priority      double precision not null default 0.5 check (priority between 0 and 1),
    progress      text,
    status        text             not null default 'active',
    last_updated  timestamptz      not null default now(),
    created_at    timestamptz      not null default now()
);

create index if not exists goals_guild_status_idx on public.goals (guild_id, status);


-- One-paragraph profile of each user, per server.
create table if not exists public.user_profiles (
    guild_id    text        not null,
    user_name   text        not null,
    user_id     text,                  -- stable Discord id; nullable so this stays additive, see v3 note above
    summary     text        not null,
    updated_at  timestamptz not null default now(),
    primary key (guild_id, user_name)
);

alter table public.user_profiles add column if not exists user_id text;

-- Lets bot.js look this row up by id once known, without touching the original name-based
-- primary key. Partial (WHERE user_id IS NOT NULL) so legacy rows that haven't been linked yet
-- -- which have user_id NULL -- never collide with each other under a plain unique index.
create unique index if not exists user_profiles_guild_user_id_idx
    on public.user_profiles (guild_id, user_id) where user_id is not null;


-- Rapport score (0 to 1) and a short "current read" of each user, per server
create table if not exists public.relationship_state (
    guild_id      text             not null,
    user_name     text             not null,
    user_id       text,                             -- stable Discord id; nullable, see v3 note above
    rapport       double precision not null default 0.5 check (rapport between 0 and 1),
    current_read  text,
    updated_at    timestamptz      not null default now(),
    primary key (guild_id, user_name)
);

alter table public.relationship_state add column if not exists user_id text;

create unique index if not exists relationship_state_guild_user_id_idx
    on public.relationship_state (guild_id, user_id) where user_id is not null;




-- v3: also returns `subject_id`, so retrieveRelevantMemories's speaker-boost can match by the
-- stable Discord id instead of only by name (a nickname change would otherwise silently stop
-- the boost from firing for that person's own memories). Same drop-then-create reasoning as v2.
drop function if exists public.match_long_term_memory(vector, text, integer);

create or replace function public.match_long_term_memory(
    query_embedding  vector(384),
    match_guild_id   text,
    match_count      integer default 15
)
returns table (
    id             bigint,
    subject        text,
    subject_id     text,
    summary        text,
    nature         text,
    importance     double precision,
    confidence     double precision,
    status         text,
    last_accessed  timestamptz,
    access_count   integer,
    similarity     double precision
)
language sql
stable
set search_path = public, extensions
as $$
    select m.id, m.subject, m.subject_id, m.summary, m.nature, m.importance, m.confidence, m.status,
           m.last_accessed, m.access_count,
           1 - (m.embedding <=> query_embedding) as similarity
    from public.long_term_memory m
    where m.guild_id = match_guild_id
      and m.status in ('active', 'fading')
      and m.embedding is not null
    order by m.embedding <=> query_embedding
    limit match_count;
$$;



-- v3: takes an optional match_subject_id, checked before the name-text match, again so a
-- nickname change doesn't stop a duplicate from being recognized as the same memory. Adding a
-- parameter changes a SQL function's identity just like changing a return column does, so this
-- also needs an explicit drop of the old 5-argument version first -- CREATE OR REPLACE would
-- otherwise leave that old version sitting alongside the new one instead of replacing it.
drop function if exists public.find_similar_memory(vector, text, text, text, double precision);

create or replace function public.find_similar_memory(
    query_embedding       vector(384),
    match_guild_id        text,
    match_nature          text,
    match_subject         text,
    match_subject_id      text default null,
    similarity_threshold  double precision default 0.86
)
returns table (
    id              bigint,
    summary         text,
    importance      double precision,
    evidence_count  integer,
    subject_id      text,
    similarity      double precision
)
language sql
stable
set search_path = public, extensions
as $$
    select m.id, m.summary, m.importance, m.evidence_count, m.subject_id,
           1 - (m.embedding <=> query_embedding) as similarity
    from public.long_term_memory m
    where m.guild_id = match_guild_id
      and m.nature = match_nature
      and (
            (match_subject_id is not null and m.subject_id = match_subject_id)
            or (
                (match_subject_id is null or m.subject_id is null)
                and lower(m.subject) is not distinct from lower(match_subject)
            )
          )
      and m.status <> 'forgotten'
      and m.embedding is not null
      and 1 - (m.embedding <=> query_embedding) >= similarity_threshold
    order by m.embedding <=> query_embedding
    limit 1;
$$;


-- Called when memories are recalled: bumps their access stats and brings a 'fading' memory back to 'active'
-- Dropped first, same as the functions above -- CREATE OR REPLACE is only guaranteed safe when
-- a function's signature genuinely never changed anywhere in its history, and an explicit drop
-- costs nothing when it hasn't, so every function in this script uses it uniformly.
drop function if exists public.reinforce_memories(bigint[]);

create or replace function public.reinforce_memories(memory_ids bigint[])
returns void
language sql
set search_path = public
as $$
    update public.long_term_memory
    set access_count  = access_count + 1,
        last_accessed = now(),
        status        = case when status = 'fading' then 'active' else status end
    where id = any (memory_ids);
$$;


-- Forgetting curve, run at the start of every reflection cycle, The bot's code calls this function but does not define the policy, so the policy lives here, Each memory gets a "fade point" that grows with its importance:
--     fade_days = 30 + 90 * importance      (importance 0.1 -> 39 days, 0.5 -> 75, 0.9 -> 111)

-- Measured from the last time the memory was recalled (or created):
--     idle >= 1x fade_days                    active   -> fading
--     idle >= 2x fade_days                    fading   -> archived
--     idle >= (4 + 4*importance)x fade_days   archived -> forgotten

-- v2: the "forgotten" multiplier used to be a flat 4x, with importance >= 0.8 permanently exempt
-- from ever reaching 'forgotten'. Combined with consolidation's +0.08-per-merge importance
-- bump, four re-observations of the same memory (easy, given how often extraction windows
-- overlap) was enough to make it immune forever. Now importance still buys a memory a much
-- longer runway -- at 0.9 importance the forgotten threshold is ~844 days idle instead of ~111
-- -- but nothing is permanently exempt.

-- Only 'active' and 'fading' memories are recalled. Nothing is ever deleted, rows marked 'archived' or 'forgotten' stay in the table. Adjust the numbers below to make Akari forget faster or slower
drop function if exists public.decay_long_term_memory(text);

create or replace function public.decay_long_term_memory(target_guild_id text)
returns void
language sql
set search_path = public
as $$
    with aged as (
        select id,
               extract(epoch from (now() - coalesce(last_accessed, created_at))) / 86400.0 as idle_days,
               30 + 90 * importance as fade_days
        from public.long_term_memory
        where guild_id = target_guild_id
          and status <> 'forgotten'
    )
    update public.long_term_memory m
    set status = case
        when m.status in ('active', 'fading', 'archived')
             and a.idle_days >= a.fade_days * (4 + 4 * m.importance)       then 'forgotten'
        when m.status in ('active', 'fading')
             and a.idle_days >= a.fade_days * 2                            then 'archived'
        when m.status = 'active'
             and a.idle_days >= a.fade_days                                then 'fading'
        else m.status
    end
    from aged a
    where a.id = m.id;
$$;


-- For each belief: how many of its supporting memories exist, and how many are still alive (active or fading). Beliefs whose evidence has mostly faded decay faster, Beliefs with no evidence links simply do not appear in the result
drop function if exists public.belief_evidence_health(bigint[]);

create or replace function public.belief_evidence_health(belief_ids bigint[])
returns table (
    belief_id    bigint,
    total_count  integer,
    alive_count  integer
)
language sql
stable
set search_path = public
as $$
    select e.belief_id,
           count(*)::integer,
           (count(*) filter (where m.status in ('active', 'fading')))::integer
    from public.belief_evidence e
    join public.long_term_memory m on m.id = e.memory_id
    where e.belief_id = any (belief_ids)
    group by e.belief_id;
$$;


-- 3. Security: server-side access only
-- Row Level Security on with no policies means only the service_role key (which bypasses RLS) can read or write, The anon and authenticated roles are cut off

alter table public.long_term_memory   enable row level security;
alter table public.beliefs            enable row level security;
alter table public.belief_evidence    enable row level security;
alter table public.goals              enable row level security;
alter table public.user_profiles      enable row level security;
alter table public.relationship_state enable row level security;

revoke all on table
    public.long_term_memory, public.beliefs, public.belief_evidence,
    public.goals, public.user_profiles, public.relationship_state
from anon, authenticated;

do $$
declare
    f record;
begin
    for f in
        select p.oid::regprocedure as signature
        from pg_proc p
        join pg_namespace n on n.oid = p.pronamespace
        where n.nspname = 'public'
          and p.proname in ('match_long_term_memory', 'find_similar_memory',
                            'reinforce_memories', 'decay_long_term_memory',
                            'belief_evidence_health')
    loop
        execute format('revoke execute on function %s from public, anon, authenticated', f.signature);
        execute format('grant execute on function %s to service_role', f.signature);
    end loop;
end
$$;

-- Make the API pick up the new tables and functions right away
notify pgrst, 'reload schema';



-- 4. Check: this should list 6 tables and 5 functions

select 'table' as kind, table_name::text as name
from information_schema.tables
where table_schema = 'public'
  and table_name in ('long_term_memory', 'beliefs', 'belief_evidence',
                     'goals', 'user_profiles', 'relationship_state')
union all
select 'function', routine_name::text
from information_schema.routines
where routine_schema = 'public'
  and routine_name in ('match_long_term_memory', 'find_similar_memory',
                       'reinforce_memories', 'decay_long_term_memory',
                       'belief_evidence_health')
order by 1, 2;
