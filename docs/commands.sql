-- Dashboard-to-runner command queue. The base table is created in Supabase;
-- this migration adds delivery state and enforces runner/admin access.

alter table public."Commands"
  add column if not exists status text not null default 'pending',
  add column if not exists result text,
  add column if not exists claimed_by uuid,
  add column if not exists created_at timestamptz not null default now(),
  add column if not exists started_at timestamptz,
  add column if not exists completed_at timestamptz;

create index if not exists commands_run_queue_idx
  on public."Commands" (run_id, status, created_at);

alter table public."Commands" enable row level security;

drop policy if exists "run participants read commands" on public."Commands";
create policy "run participants read commands"
  on public."Commands" for select
  using (
    exists (
      select 1
      from public."Debug_Sessions" ds
      where ds.id = "Commands".run_id
        and ds.project_id = "Commands".project_id
        and (
          ds.user_id = auth.uid()
          or exists (
            select 1
            from public."Projects" p
            join public."Teams" t on t.team_id = p.team_id
            where p.project_id = "Commands".project_id
              and auth.uid() = any (t.admin_ids)
          )
        )
    )
  );

drop policy if exists "run participants enqueue commands" on public."Commands";
create policy "run participants enqueue commands"
  on public."Commands" for insert
  with check (
    user_id = auth.uid()
    and status = 'pending'
    and exists (
      select 1
      from public."Debug_Sessions" ds
      where ds.id = "Commands".run_id
        and ds.project_id = "Commands".project_id
        and (
          ds.user_id = auth.uid()
          or exists (
            select 1
            from public."Projects" p
            join public."Teams" t on t.team_id = p.team_id
            where p.project_id = "Commands".project_id
              and auth.uid() = any (t.admin_ids)
          )
        )
    )
  );

drop policy if exists "run participants update commands" on public."Commands";
create policy "run participants update commands"
  on public."Commands" for update
  using (
    exists (
      select 1
      from public."Debug_Sessions" ds
      where ds.id = "Commands".run_id
        and ds.project_id = "Commands".project_id
        and (
          ds.user_id = auth.uid()
          or exists (
            select 1
            from public."Projects" p
            join public."Teams" t on t.team_id = p.team_id
            where p.project_id = "Commands".project_id
              and auth.uid() = any (t.admin_ids)
          )
        )
    )
  )
  with check (
    status in ('pending', 'processing', 'completed', 'failed')
    and exists (
      select 1
      from public."Debug_Sessions" ds
      where ds.id = "Commands".run_id
        and ds.project_id = "Commands".project_id
        and (
          ds.user_id = auth.uid()
          or exists (
            select 1
            from public."Projects" p
            join public."Teams" t on t.team_id = p.team_id
            where p.project_id = "Commands".project_id
              and auth.uid() = any (t.admin_ids)
          )
        )
    )
  );