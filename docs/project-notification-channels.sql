-- Per-project notification channels (workspace -> project -> channel).
--
-- The workspace keeps ONE Discord connection (team_integrations) and a default
-- channel. This table lets each project override that channel, or be muted.
-- A project with no row uses the workspace default. Run once in the Supabase
-- SQL editor.

create table if not exists public.team_integrations (
  project_id           uuid primary key references public."Projects" (project_id) on delete cascade,
  team_id              uuid not null references public."Teams" (team_id) on delete cascade,
  discord_channel_id   text,
  discord_channel_name text,
  muted                boolean not null default false,
  updated_at           timestamptz not null default now()
);

create index if not exists team_integrations_team_id_idx
  on public.team_integrations (team_id);

alter table public.team_integrations enable row level security;

-- Every workspace member can read the routing (the dashboard needs it to send alerts).
create policy "members read project channels"
  on public.team_integrations for select
  using (
    exists (
      select 1 from public."Teams" t
      where t.team_id = team_integrations.team_id
        and auth.uid()::text = any (t.members)
    )
  );

-- Only workspace admins change it.
create policy "admins write project channels"
  on public.team_integrations for all
  using (
    exists (
      select 1 from public."Teams" t
      where t.team_id = team_integrations.team_id
        and auth.uid() = any (t.admin_ids)
    )
  )
  with check (
    exists (
      select 1 from public."Teams" t
      where t.team_id = team_integrations.team_id
        and auth.uid() = any (t.admin_ids)
    )
  );

-- The `send-discord-nonification` edge function must also honour the new request fields
-- (see below), otherwise every alert still lands in the workspace default channel:
--   body: { team_id, project_id, channel_id, title, text }
--   * if channel_id is present, post there -- after checking it belongs to the team's
--     connected Discord server (never trust the browser's channel id blindly);
--   * otherwise fall back to team_integrations.discord_channel_id.
