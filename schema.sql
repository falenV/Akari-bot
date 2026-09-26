-- Akari schema 
-- How to use: open your Supabase project > SQL Editor > New query, paste this whole file, and press Run. It is safe to run more than once, including over a database that already ran an earlier version of this script, it only creates or alters what's missing/changed and never deletes data


create extension if not exists vector with schema extensions;




create table if not exists public.long_term_memory (
    id              bigint generated always as identity primary key,
    guild_id        text             not null,
    subject         text,                                  
    subject_id      text,                                   
    summary         text             not null,              
    nature          text             not null default 'fact', 
    importance      double precision not null default 0.5 check (importance between 0 and 1),
    confidence      double precision not null default 0.7 check (confidence between 0 and 1),
    embedding       vector(384),
    embedding_model text,                                  
    status          text             not null default 'active'
                        check (status in ('active', 'fading', 'archived', 'forgotten')),
    evidence_count  integer          not null default 1,    
    access_count    integer          not null default 0,    
    last_accessed   timestamptz      not null default now(),
    created_at      timestamptz      not null default now()
);


alter table public.long_term_memory add column if not exists embedding_model text;
alter table public.long_term_memory add column if not exists subject_id text;

create index if not exists long_term_memory_guild_status_idx
    on public.long_term_memory (guild_id, status, created_at desc);

create index if not exists long_term_memory_subject_id_idx
    on public.long_term_memory (guild_id, subject_id) where subject_id is not null;



create table if not exists public.beliefs (
    id              bigint generated always as identity primary key,
    guild_id        text             not null,
    scope           text             not null default 'user', 
    subject         text,                                     
    subject_id      text,                                     
    statement       text             not null,
    confidence      double precision not null default 0.45 check (confidence between 0 and 1),
    evidence_count  integer          not null default 1,
    embedding       vector(384),
    embedding_model text,                                     
    last_updated    timestamptz      not null default now(),
    created_at      timestamptz      not null default now()
);


alter table public.beliefs add column if not exists embedding_model text;
alter table public.beliefs add column if not exists subject_id text;

create index if not exists beliefs_guild_idx on public.beliefs (guild_id);

create index if not exists beliefs_subject_id_idx
    on public.beliefs (guild_id, subject_id) where subject_id is not null;




create table if not exists public.belief_evidence (
    belief_id  bigint not null references public.beliefs (id) on delete cascade,
    memory_id  bigint not null,
    primary key (belief_id, memory_id)
);

create index if not exists belief_evidence_memory_idx on public.belief_evidence (memory_id);

 
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


-- One-paragraph profile of each user, per server
create table if not exists public.user_profiles (
    guild_id    text        not null,
    user_name   text        not null,
    user_id     text,                  
    summary     text        not null,
    updated_at  timestamptz not null default now(),
    primary key (guild_id, user_name)
);

alter table public.user_profiles add column if not exists user_id text;


create unique index if not exists user_profiles_guild_user_id_idx
    on public.user_profiles (guild_id, user_id) where user_id is not null;



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


drop function if exists public.match_long_term_memory(vector, text, integer);

create function public.match_long_term_memory(
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




drop function if exists public.find_similar_memory(vector, text, text, text, double precision);

create function public.find_similar_memory(
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
