"use strict";

/* ============================================================
   PULSE TEAMS
   Supabase + dashboard logic
============================================================ */

const SUPABASE_URL =
  "https://wlvcvbaoxqgryarasjvw.supabase.co";

const SUPABASE_KEY =
  "sb_publishable_o1djx0_qUWbPrYZXu9XiVw_PU1Wdx6U";

const LS_EMAIL_KEY =
  "pulse-dashboard:last-email";

const SS_SESSION_KEY =
  "pulse-dashboard:session";

const CODEC_PREFIX = "PZ1:";

const LIVE_WINDOW_MS = 5 * 60 * 1000;

const LS_NOTIF_KEY =
  "pulse-dashboard:notification-settings";

const LS_PINNED_RUNS_PREFIX =
  "pulse-dashboard:pinned-runs:";
const LS_UNPINNED_LIVE_RUNS_PREFIX =
  "pulse-dashboard:unpinned-live-runs:";
const LS_SIDEBAR_COLLAPSED_KEY =
  "pulse-dashboard:sidebar-collapsed";

const SS_PENDING_OAUTH_TEAM_KEY =
  "pulse-dashboard:pending-oauth-team";

// Fill these in with your own app's OAuth client IDs (public values --
// safe to ship in the frontend; only the matching client *secret* is
// confidential, and that lives solely in the slack-oauth-callback edge
// function's env, never here). See NOTIFICATIONS_SETUP.md.
const DISCORD_CLIENT_ID = "1553215834871177296";
const SLACK_CLIENT_ID = "REPLACE_WITH_YOUR_SLACK_CLIENT_ID";

// Discord permission bitmask: View Channel (1024) + Send Messages (2048).
const DISCORD_BOT_PERMISSIONS = "3072";

// Shared identity so Discord messages are recognizably "Pulse" -- its
// name/avatar are whatever you set once for the bot in the Developer
// Portal, applied automatically to every message the bot posts.
const PULSE_BOT_NAME = "Pulse";
const PULSE_BOT_ICON_DATA_URI =
  "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Crect width='100' height='100' rx='22' fill='%231d1d1f'/%3E%3Ctext x='50' y='70' font-family='Arial' font-weight='700' font-size='62' fill='white' text-anchor='middle'%3EP%3C/text%3E%3C/svg%3E";

let currentAccessToken = null;
let currentRefreshToken = null;
let refreshTokenPromise = null;
let currentUser = null;
let currentTeam = null;
let currentProject = null;
let currentSessions = [];
let workspaceRecords = new Map();
let sidebarContextTeam = null;

let activeSessionId = null;
let autoSelectLiveRun = false;
let pinnedRunIds = new Set();
let unpinnedLiveRunIds = new Set();
let observedLiveRunIds = new Set();
let pinnedRunsProjectId = null;
let commandDrafts = new Map();
let commandSuggestionIndex = 0;

let emailCache = new Map();

let refreshInFlight = false;
// Bumped by every new load of a list; a load that finds it changed after an await is stale
// (another workspace/project was opened meanwhile) and drops what it fetched.
let viewGeneration = 0;
let refreshGeneration = 0;
let lastRenderedSignature = "";

// Only the browser-push toggle lives in localStorage now -- Discord/Slack
// connections are shared per-workspace, stored in team_integrations.
let notificationSettings = {
  browserPush: false
};

// { discord_guild_id, discord_guild_name, discord_channel_id,
//   discord_channel_name, slack_team_name, slack_channel_name,
//   slack_webhook_url, ... } for the currently open workspace, or null.
let teamIntegration = null;

// Snapshot of "live" / "incident" state as of the last poll, per workspace.
// null means "no baseline yet" -- the next refresh seeds it silently instead
// of firing notifications for things that were already true when the
// dashboard was opened.
let notifyBaseline = null;

// Set right before opening the notifications modal after an OAuth redirect
// brings the user back, so the modal can show "Discord connected." once.
let pendingIntegrationNotice = null;

// What a prompt sent from here can be: commands about the machine's terminal or its other
// runs (/exit, /mouse, /copy, /config, /agent, /monitor, /change ...) are refused by the runner.
const HOME_COMMANDS = [
  ["/run", "start a script under Pulse and watch it: /run train.py --epochs 3"],
  ["/files", "what the agent sees in full, and how much it can search"],
  ["/add", "put files or folders in focus"],
  ["/drop", "take a file out of focus"],
  ["/review", "show a diff and ask before applying: /review on|off"],
  ["/undo", "undo the latest change"],
  ["/log", "the change history"],
  ["/help", "every command"]
];

const DEBUG_COMMANDS = [
  ["/experiment", "run isolated proxy experiments: /experiment experiments/spec.json"],
  ["/findings", "what the detectors currently believe, worst first"],
  ["/curve", "one value's history: /curve val_loss"],
  ["/vars", "every value being tracked"],
  ["/audit", "have the agent audit the whole run now"],
  ["/audits", "scheduled audits by the agent: /audits on|off"],
  ["/trace", "what feeds a variable and what it feeds: /trace loss"],
  ["/source", "show the training script"],
  ["/pause", "pause the run"],
  ["/resume", "resume a paused run"],
  ["/stop", "stop the training run"],
  ["/restart", "run the script again with the current code"],
  ["/output", "the last lines the run printed"],
  ["/interval", "sample faster or slower: /interval 0.5"],
  ["/quiet", "stop announcing findings (/loud resumes)"],
  ["/home", "back to the agent for anything else; the run stays watched"],
  ["/close", "stop watching the open run"]
];


/* ============================================================
   ELEMENTS
============================================================ */

const els = {
  viewLogin: document.getElementById("view-login"),
  viewDashboard: document.getElementById("view-dashboard"),
  workspaceSidebar: document.getElementById("workspace-sidebar"),
  sidebarToggle: document.getElementById("sidebar-toggle"),
  sidebarContextMenu: document.getElementById("sidebar-context-menu"),

  loginForm: document.getElementById("login-form"),
  loginUsername: document.getElementById("login-username"),
  loginPassword: document.getElementById("login-password"),
  loginError: document.getElementById("login-error"),

  homeBtn: document.getElementById("home-btn"),
  topbarRight: document.getElementById("topbar-right"),
  topbarWorkspace: document.getElementById("topbar-workspace"),
  footerAccount: document.getElementById("footer-account"),
  footerUser: document.getElementById("footer-user"),
  footerSignOut: document.getElementById("footer-sign-out"),

  workspaceList: document.getElementById("workspace-list"),
  workspacesEmpty: document.getElementById("workspaces-empty"),
  projectsWorkspace: document.getElementById("projects-workspace"),

  workspaceTitle: document.getElementById("workspace-title"),

  projectList: document.getElementById("project-list"),
  projectsEmpty: document.getElementById("projects-empty"),
  projectsWorkspace: document.getElementById("projects-workspace"),
  projectJoinForm: document.getElementById("project-join-form"),
  projectJoinCode: document.getElementById("project-join-code"),
  projectJoinMsg: document.getElementById("project-join-msg"),
  projectCrumb: document.getElementById("project-crumb"),

  runCount: document.getElementById("run-count"),

  sessionList: document.getElementById("session-list"),
  emptyState: document.getElementById("empty-state"),
  activeConsole: document.getElementById("active-console"),

  refreshBtn: document.getElementById("refresh-btn"),
  refreshLabel: document.getElementById("refresh-label"),

  notificationsBtn: document.getElementById("notifications-btn"),
  notificationsModal: document.getElementById("notifications-modal"),
  notificationsCloseBtn: document.getElementById("notifications-close-btn"),
  notificationsSaveBtn: document.getElementById("notifications-save-btn"),
  notificationsTestBtn: document.getElementById("notifications-test-btn"),
  notifBrowserPush: document.getElementById("notif-browser-push"),
  notifPushStatus: document.getElementById("notif-push-status"),
  notifSaveStatus: document.getElementById("notif-save-status")
};


/* ============================================================
   SUPABASE GET
============================================================ */

// Every request to Supabase. An access token lasts an hour: when the server says it has
// expired (401), it is renewed once with the refresh token and the request sent again; if
// that is refused too, the sign-in is over and the login screen says so.
async function supabaseFetch(url, options = {}) {

  const send = () => fetch(url, {
    ...options,
    headers: {
      ...(options.headers || {}),
      apikey: SUPABASE_KEY,
      Authorization: `Bearer ${currentAccessToken || SUPABASE_KEY}`
    }
  });

  let response = await send();

  if (response.status === 401 && currentAccessToken) {
    if (await refreshAccessToken()) {
      response = await send();
    } else {
      sessionExpired();
    }
  }

  return response;
}


function refreshAccessToken() {

  if (!currentRefreshToken) {
    return Promise.resolve(false);
  }

  if (!refreshTokenPromise) {
    refreshTokenPromise = (async () => {
      try {
        const response = await fetch(
          `${SUPABASE_URL}/auth/v1/token?grant_type=refresh_token`,
          {
            method: "POST",
            headers: {
              apikey: SUPABASE_KEY,
              Authorization: `Bearer ${SUPABASE_KEY}`,
              "Content-Type": "application/json"
            },
            body: JSON.stringify({ refresh_token: currentRefreshToken })
          }
        );
        if (!response.ok) return false;
        const data = await response.json();
        if (!data.access_token) return false;
        currentAccessToken = data.access_token;
        currentRefreshToken = data.refresh_token || currentRefreshToken;
        if (currentUser) {
          currentUser.access_token = currentAccessToken;
          currentUser.refresh_token = currentRefreshToken;
          sessionStorage.setItem(SS_SESSION_KEY, JSON.stringify(currentUser));
        }
        return true;
      } catch (error) {
        console.warn("Could not renew the sign-in:", error);
        return false;
      } finally {
        refreshTokenPromise = null;
      }
    })();
  }

  return refreshTokenPromise;
}


function sessionExpired() {

  if (!currentUser) return;
  signOut();
  els.loginError.textContent = "Your sign-in expired. Sign in again.";
}


async function pgGet(table, params = {}) {

  const url =
    new URL(`${SUPABASE_URL}/rest/v1/${table}`);

  Object.entries(params).forEach(([key, value]) => {
    url.searchParams.set(key, value);
  });

  const response = await supabaseFetch(url.toString());

  if (!response.ok) {

    const text =
      await response.text().catch(() => "");

    throw new Error(
      `GET ${table} -> HTTP ${response.status}: ${text}`
    );
  }

  return response.json();
}


// `expectRows`: the change must reach a row. Row Level Security turns a change the user
// may not make into "0 rows updated", which is otherwise a silent success.
async function pgPatch(table, match, body, { expectRows = false } = {}) {

  const url =
    new URL(`${SUPABASE_URL}/rest/v1/${table}`);

  Object.entries(match).forEach(([key, value]) => {
    url.searchParams.set(key, value);
  });

  const response = await supabaseFetch(url.toString(), {
    method: "PATCH",

    headers: {
      "Content-Type": "application/json",
      Prefer: expectRows ? "return=representation" : "return=minimal"
    },

    body: JSON.stringify(body)
  });

  if (!response.ok) {

    const text =
      await response.text().catch(() => "");

    throw new Error(
      `PATCH ${table} -> HTTP ${response.status}: ${text}`
    );
  }

  if (expectRows) {
    const rows = await response.json().catch(() => []);
    if (!Array.isArray(rows) || !rows.length) {
      throw new Error("Only a workspace admin can change that.");
    }
  }
}


async function pgPost(table, body) {

  const response = await supabaseFetch(
    `${SUPABASE_URL}/rest/v1/${table}`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Prefer: "return=minimal"
      },
      body: JSON.stringify(body)
    }
  );

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new Error(`POST ${table} -> HTTP ${response.status}: ${text}`);
  }
}


async function callEdgeFunction(name, { method = "POST", params, body } = {}) {

  const url =
    new URL(`${SUPABASE_URL}/functions/v1/${name}`);

  if (params) {

    Object.entries(params).forEach(([key, value]) => {
      url.searchParams.set(key, value);
    });
  }

  const response = await supabaseFetch(url.toString(), {
    method,

    headers: {
      "Content-Type": "application/json"
    },

    body: body ? JSON.stringify(body) : undefined
  });

  const payload =
    await response.json().catch(() => ({}));

  if (!response.ok) {

    throw new Error(
      payload.error || `${name} -> HTTP ${response.status}`
    );
  }

  return payload;
}


/* ============================================================
   DECOMPRESSION
============================================================ */

function base64ToBytes(base64) {

  const binary = atob(base64);

  const bytes =
    new Uint8Array(binary.length);

  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }

  return bytes;
}


async function decodeEntry(value) {

  if (typeof value !== "string") {
    return value;
  }

  if (!value.startsWith(CODEC_PREFIX)) {

    try {
      return JSON.parse(value);
    } catch {
      return value;
    }
  }

  try {

    const compressed =
      base64ToBytes(
        value.slice(CODEC_PREFIX.length)
      );

    const stream =
      new Blob([compressed])
        .stream()
        .pipeThrough(
          new DecompressionStream("deflate")
        );

    const buffer =
      await new Response(stream).arrayBuffer();

    const text =
      new TextDecoder().decode(buffer);

    return JSON.parse(text);

  } catch (error) {

    console.warn(
      "Failed to decode Pulse entry:",
      error
    );

    return null;
  }
}


async function decodeEntries(values) {

  if (!Array.isArray(values)) {
    return [];
  }

  const decoded =
    await Promise.all(
      values.map(decodeEntry)
    );

  return decoded.filter(
    value =>
      value !== null &&
      value !== undefined
  );
}


/* ============================================================
   AUTH
============================================================ */

async function loginUser(email, password) {

  try {

    const response = await fetch(
      `${SUPABASE_URL}/auth/v1/token?grant_type=password`,
      {
        method: "POST",

        headers: {
          apikey: SUPABASE_KEY,
          Authorization:
            `Bearer ${SUPABASE_KEY}`,
          "Content-Type":
            "application/json"
        },

        body: JSON.stringify({
          email,
          password
        })
      }
    );

    if (!response.ok) {
      return null;
    }

    const authData =
      await response.json();

    if (
      !authData.access_token ||
      !authData.user
    ) {
      return null;
    }

    currentAccessToken =
      authData.access_token;

    currentRefreshToken =
      authData.refresh_token || null;

    const profileRows =
      await pgGet("profiles", {
        id: `eq.${authData.user.id}`,
        select: "id,email,personal_plan"
      });

    const profile = profileRows[0];

    if (!profile) {
      return null;
    }

    return {
      id: profile.id,
      email: profile.email,
      access_token:
        authData.access_token,
      refresh_token:
        authData.refresh_token || null
    };

  } catch (error) {

    console.error(
      "Authentication error:",
      error
    );

    return null;
  }
}


/* ============================================================
   WORKSPACES
============================================================ */

const TEAM_SELECT =
  "team_id,members,admin_ids,join_code,plan,owner_id,name";

const TEAM_SELECT_NO_NAME =
  "team_id,members,admin_ids,join_code,plan,owner_id";

// Mirrors pulse_supabase.py's _is_unknown_column_error -- lets a
// deployment that hasn't run the `name` migration yet degrade instead
// of failing to load workspaces at all.
function isUnknownColumnError(error) {

  const message =
    String(error?.message || error || "");

  if (message.includes("PGRST204")) {

    return true;
  }

  const lower =
    message.toLowerCase();

  return (
    lower.includes("column") &&
    (
      lower.includes("does not exist") ||
      lower.includes("not found")
    )
  );
}

async function loadWorkspacesForUser(userId) {

  try {

    return await pgGet("Teams", {
      members: `cs.{${userId}}`,
      select: TEAM_SELECT
    });

  } catch (error) {

    if (!isUnknownColumnError(error)) {

      throw error;
    }

    // This deployment's Teams table doesn't have `name` yet -- retry
    // without it rather than failing the whole workspace list.
    return pgGet("Teams", {
      members: `cs.{${userId}}`,
      select: TEAM_SELECT_NO_NAME
    });
  }
}


/* ============================================================
   PROJECTS  (workspace -> project -> run)

   A normal project is visible to everyone in the workspace. A
   secret project is visible to the workspace's admins and to the
   members who joined it with its join code -- the same rule the
   Pulse CLI uses. Mirror it in Row Level Security for it to be a
   real boundary; the filtering here only shapes what the page shows.
============================================================ */

const PROJECT_SELECT =
  "project_id,team_id,name,repo,is_secret,members,created_at";


function isTeamAdmin(team) {

  return (team.admin_ids || [])
    .includes(currentUser.id);
}


function projectVisible(project, team) {

  if (!project.is_secret) {
    return true;
  }

  return (
    isTeamAdmin(team) ||
    (project.members || [])
      .includes(currentUser.id)
  );
}


async function loadProjectsForTeam(team) {

  const params = {
    team_id: `eq.${team.team_id}`,
    select: PROJECT_SELECT,
    order: "created_at.asc"
  };

  if (!isTeamAdmin(team)) {

    params.or =
      `(is_secret.is.null,is_secret.eq.false,members.cs.{${currentUser.id}})`;
  }

  const rows =
    await pgGet("Projects", params);

  return rows.filter(
    project =>
      projectVisible(project, team)
  );
}


async function joinSecretProject(team, code) {

  const rows =
    await pgGet("Projects", {
      team_id: `eq.${team.team_id}`,
      secret_join_code: `eq.${code}`,
      is_secret: "eq.true",
      select: PROJECT_SELECT
    });

  const project = rows[0];

  if (!project) {

    throw new Error(
      "No secret project in this workspace has that join code."
    );
  }

  const members = project.members || [];

  if (!members.includes(currentUser.id)) {

    members.push(currentUser.id);

    await pgPatch(
      "Projects",
      { project_id: `eq.${project.project_id}` },
      { members }
    );

    project.members = members;
  }

  return project;
}


function getProjectName(project) {

  return (
    (project.name || "").trim() ||
    "Untitled project"
  );
}


function getRepoName(repo) {

  if (!repo || repo === "unknown") {
    return "";
  }

  return repo
    .split("/")
    .slice(-1)[0]
    .replace(/\.git$/, "");
}


/* ============================================================
   SESSIONS
============================================================ */

function latestEnvInfo(telemetry) {

  for (
    let index = telemetry.length - 1;
    index >= 0;
    index--
  ) {

    const entry = telemetry[index];

    if (
      entry &&
      (entry.type === "env_info" || entry.script || entry.gpu_name || entry.python_version)
    ) {
      return entry;
    }
  }

  return null;
}


async function fetchUserEmails(userIds) {

  const missing =
    userIds.filter(
      id =>
        id &&
        !emailCache.has(id)
    );

  if (missing.length) {

    try {

      const rows =
        await pgGet("profiles", {
          id: `in.(${missing.join(",")})`,
          select: "id,email"
        });

      rows.forEach(row => {
        emailCache.set(
          row.id,
          row.email
        );
      });

    } catch (error) {

      console.warn(
        "Could not resolve user emails:",
        error
      );
    }
  }

  return userIds.map(
    id =>
      emailCache.get(id) || null
  );
}


// `projectIds`: one project id, or an array of them (a workspace's live count).
async function loadSessions(projectIds) {

  const ids =
    Array.isArray(projectIds)
      ? projectIds
      : [projectIds];

  if (!ids.length) {
    return [];
  }

  const projectFilter =
    ids.length === 1
      ? `eq.${ids[0]}`
      : `in.(${ids.join(",")})`;

  let rows;

  try {

    rows =
      await pgGet("Debug_Sessions", {
        project_id: projectFilter,

        select:
          [
            "id",
            "project_id",
            "created_at",
            "user_id",
            "git_commit_sha",
            "agent_logs",
            "error_tracebacks",
            "telemetry",
            "incidents",
            "uptime_seconds",
            "downtime_seconds"
          ].join(","),

        order: "created_at.desc",
        limit: "50"
      });

  } catch (error) {

    console.warn(
      "Falling back to minimal session query:",
      error
    );

    rows =
      await pgGet("Debug_Sessions", {
        project_id: projectFilter,

        select:
          "id,project_id,created_at,user_id,git_commit_sha,telemetry",

        order: "created_at.desc",
        limit: "50"
      });
  }

  const decoded =
    await Promise.all(
      rows.map(async row => {

        const telemetry =
          await decodeEntries(
            row.telemetry
          );

        const incidents =
          await decodeEntries(
            row.incidents || []
          );

        const agentLogs =
          await decodeEntries(
            row.agent_logs || []
          );

        const errorTracebacks =
          await decodeEntries(
            row.error_tracebacks || []
          );

        return {
          ...row,

          telemetry,

          incidents:
            incidents.sort(
              (a, b) =>
                (b.t || 0) -
                (a.t || 0)
            ),

          agentLogs,
          errorTracebacks,

          env:
            latestEnvInfo(telemetry)
        };
      })
    );

  const emails =
    await fetchUserEmails(
      decoded.map(
        session => session.user_id
      )
    );

  decoded.forEach(
    (session, index) => {
      session.username =
        emails[index];
    }
  );

  return decoded;
}


async function loadCommands(sessions) {

  const ids = sessions.map(session => session.id).filter(Boolean);
  if (!ids.length) return;

  try {
    const rows = await pgGet("Commands", {
      run_id: `in.(${ids.join(",")})`,
      select: "id,run_id,user_id,command,status,result,created_at,completed_at",
      order: "created_at.desc",
      limit: "200"
    });
    const byRun = new Map();
    rows.reverse().forEach(row => {
      const commands = byRun.get(row.run_id) || [];
      commands.push(row);
      byRun.set(row.run_id, commands);
    });
    sessions.forEach(session => {
      session.commands = byRun.get(session.id) || [];
    });
  } catch (error) {
    console.warn("Could not load run commands:", error);
    sessions.forEach(session => {
      session.commands = [];
    });
  }
}


// A question Pulse asked on the machine (apply this change? run this command?) is a Commands
// row of its own, "pulse:ask {json}", left "processing" until someone answers -- here, or at
// the machine. Answering here completes the row with the answer.
const QUESTION_PREFIX = "pulse:ask ";

function questionOf(command) {
  const text = String(command?.command || "");
  if (!text.startsWith(QUESTION_PREFIX)) return null;
  try {
    const parsed = JSON.parse(text.slice(QUESTION_PREFIX.length));
    return {
      label: String(parsed.label || "Pulse is asking:"),
      options: Array.isArray(parsed.options) ? parsed.options.map(String) : null,
      detail: parsed.detail ? String(parsed.detail) : "",
      context: parsed.context ? String(parsed.context) : ""
    };
  } catch {
    return { label: text.slice(QUESTION_PREFIX.length), options: null, detail: "", context: "" };
  }
}

// What Pulse is doing right now on the machine: a row of its own, "pulse:live", whose result
// the runner rewrites about once a second (the agent's reasoning and tool calls as they
// stream, what is running, the run's step and values) and at least every 15 s.
const LIVE_PREFIX = "pulse:live";
const LIVE_FRESH_SECONDS = 45;

function isLiveRow(command) {
  return String(command?.command || "").startsWith(LIVE_PREFIX);
}

function isRunnerRow(command) {
  return isLiveRow(command) || Boolean(questionOf(command));
}

// The newest live state of a session that is still fresh, or null.
function liveStateOf(session) {
  const row = (session.commands || []).filter(command => isLiveRow(command) && command.status === "processing").at(-1);
  if (!row || !row.result) return null;
  try {
    const state = JSON.parse(row.result);
    if (!state || state.ended || Date.now() / 1000 - Number(state.t || 0) > LIVE_FRESH_SECONDS) return null;
    return state;
  } catch {
    return null;
  }
}

// "Apply this change? [y/N]" and the like: Yes / No buttons send what the terminal takes.
function isYesNo(label) {
  return /\[\s*y\s*\/\s*n\s*\]|\(\s*y\s*\/\s*n\s*\)/i.test(label);
}

function displayQuestionLabel(label) {
  return String(label || "")
    .replace(/\s*(?:\[\s*y\s*\/\s*n\s*\]|\(\s*y\s*\/\s*n\s*\))\s*$/i, "")
    .trim();
}

async function answerQuestion(commandId, answer) {
  await pgPatch(
    "Commands",
    { id: `eq.${commandId}`, status: "eq.processing" },
    { status: "completed", result: answer, completed_at: new Date().toISOString() },
    { expectRows: true }
  );
}


/* ============================================================
   FORMATTING
============================================================ */

// Also used inside attribute values, so quotes are escaped too.
function escapeHtml(value) {

  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

const CLI_ACTION_NAMES = new Set([
  "AMPSTATUS", "CALC", "CALLERS", "CHECK_SHAPE", "CORR", "CREATE", "DEP_GRAPH",
  "DIFFSTATS", "DOC_LOOKUP", "DRYRUN", "EDIT", "EDIT_FILE", "FIND_DEFINITION",
  "FIND_REFERENCES", "GPUSTATUS", "GREP", "GRADCHECK", "HARDEXAMPLES", "HISTOGRAM",
  "LAYERSTATS", "LIST_FILES", "MLLINT", "OUTLIER", "OUTLINE", "PASTFIX", "READ_FILE",
  "REPL", "REPLACE_SYMBOL", "REPLAY", "RESTART", "RESTART_RUN", "ROLLBACK", "RUN",
  "RUNCOMPARE", "RUN_COMMAND", "RUN_STATUS", "RUNSTATUS", "SEEDCHECK", "SHAPETRACE",
  "SMOKE_TEST", "STOP", "STOP_RUN", "TERMINAL", "TODO_WRITE", "TRACE", "TRACE_VARIABLE",
  "VIEW", "WRITE_FILE"
]);

function renderCliText(value) {
  const text = String(value ?? "");
  const ansi = /\u001b\[([0-9;]*)m/g;
  const result = [];
  let active = new Set();
  let position = 0;
  let match;

  const appendRaw = segment => {
    if (!segment) return;
    const classes = [...active];
    result.push(classes.length
      ? `<span class="${classes.join(" ")}">${escapeHtml(segment)}</span>`
      : escapeHtml(segment));
  };
  const append = segment => {
    segment.split(/(\n)/).forEach((line, index) => {
      if (index % 2) {
        appendRaw(line);
        return;
      }
      const action = /^(\s*)([A-Z][A-Z0-9_]{2,})(:)(?=\s|$)/.exec(line);
      if (action && CLI_ACTION_NAMES.has(action[2])) {
        appendRaw(action[1]);
        result.push(`<span class="cli-action-name">${escapeHtml(action[2] + action[3])}</span>`);
        appendRaw(line.slice(action[0].length));
      } else {
        appendRaw(line);
      }
    });
  };

  while ((match = ansi.exec(text))) {
    append(text.slice(position, match.index));

    const codes = (match[1] || "0").split(";").map(code => Number(code || 0));
    for (let index = 0; index < codes.length; index += 1) {
      let code = codes[index];
      let extendedColor = false;
      if (code === 38 && codes[index + 1] === 5 && codes[index + 2] !== undefined) {
        code = codes[index + 2];
        extendedColor = true;
        index += 2;
      }
      if (code === 0) active = new Set();
      else if (code === 1) active.add("cli-ansi-bold");
      else if (code === 2) active.add("cli-ansi-dim");
      else if (code === 3) active.add("cli-ansi-italic");
      else if (code === 22) {
        active.delete("cli-ansi-bold");
        active.delete("cli-ansi-dim");
      } else if (code === 23) active.delete("cli-ansi-italic");
      else if ([31, 91].includes(code)) {
        active.delete("cli-ansi-orange");
        active.delete("cli-ansi-blue");
        active.add("cli-ansi-red");
      } else if ((!extendedColor && [33, 93].includes(code)) ||
                 (extendedColor && [208, 214].includes(code))) {
        active.delete("cli-ansi-red");
        active.delete("cli-ansi-blue");
        active.add("cli-ansi-orange");
      } else if ((!extendedColor && [36, 96].includes(code)) ||
                 (extendedColor && [33, 39].includes(code))) {
        active.delete("cli-ansi-red");
        active.delete("cli-ansi-orange");
        active.add("cli-ansi-blue");
      } else if (code === 39) {
        active.delete("cli-ansi-red");
        active.delete("cli-ansi-orange");
        active.delete("cli-ansi-blue");
      }
    }
    position = ansi.lastIndex;
  }

  append(text.slice(position));
  return result.join("");
}


function fmtDuration(seconds) {

  if (!seconds || seconds < 1) {
    return "0m";
  }

  const hours =
    Math.floor(seconds / 3600);

  const minutes =
    Math.floor(
      (seconds % 3600) / 60
    );

  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  }

  return `${minutes}m`;
}


function fmtTrackedTime(seconds) {

  if (!seconds || seconds < 1) {
    return "0m";
  }

  const hours =
    Math.floor(seconds / 3600);

  const minutes =
    Math.floor(
      (seconds % 3600) / 60
    );

  const secs =
    Math.floor(seconds % 60);

  if (hours) {
    return `${hours}h ${minutes}m`;
  }

  if (minutes) {
    return `${minutes}m ${secs}s`;
  }

  return `${secs}s`;
}


function fmtRelativeTime(seconds) {

  if (!seconds) {
    return "unknown";
  }

  const diff =
    Date.now() -
    seconds * 1000;

  const minutes =
    Math.round(
      diff / 60000
    );

  if (minutes < 1) {
    return "just now";
  }

  if (minutes < 60) {
    return `${minutes}m ago`;
  }

  const hours =
    Math.round(minutes / 60);

  if (hours < 24) {
    return `${hours}h ago`;
  }

  return `${Math.round(hours / 24)}d ago`;
}


/* ============================================================
   SESSION STATE
============================================================ */

function isLive(session) {

  if (liveStateOf(session)) {
    return true;
  }

  const timestamps = [

    ...session.telemetry.map(
      entry => entry.t || 0
    ),

    ...session.incidents.map(
      entry => entry.t || 0
    ),

    ...(session.agentLogs || []).map(
      entry => entry.t || 0
    )
  ];

  if (!timestamps.length) {
    return false;
  }

  const lastActivity =
    Math.max(...timestamps) * 1000;

  return (
    lastActivity > 0 &&
    Date.now() - lastActivity <
      LIVE_WINDOW_MS
  );
}


function latestTelemetry(session) {

  if (
    !session.telemetry ||
    !session.telemetry.length
  ) {
    return {};
  }

  return (
    session.telemetry[
      session.telemetry.length - 1
    ] || {}
  );
}


function findMetric(session, names) {

  const telemetry =
    session.telemetry || [];

  for (
    let i = telemetry.length - 1;
    i >= 0;
    i--
  ) {

    const entry = telemetry[i];

    if (!entry) continue;

    for (const name of names) {

      if (
        entry[name] !== undefined &&
        entry[name] !== null
      ) {
        return entry[name];
      }
    }
  }

  return null;
}


function metricValue(value) {

  if (
    value === null ||
    value === undefined
  ) {
    return "—";
  }

  if (
    typeof value === "number"
  ) {

    if (!Number.isFinite(value)) {
      return "NaN";
    }

    if (
      Math.abs(value) < 0.001 ||
      Math.abs(value) >= 10000
    ) {
      return value.toExponential(3);
    }

    return Number(value.toFixed(4))
      .toString();
  }

  return String(value);
}


/* ============================================================
   TOPBAR
============================================================ */

function renderTopbarRight() {

  if (!currentUser) {

    els.topbarRight.innerHTML = "";
    els.topbarWorkspace.textContent = "";
    els.footerAccount.hidden = true;
    document.body.classList.remove("dashboard-active");
    document.body.classList.add("login-active");

    return;
  }

  els.topbarRight.innerHTML = "";
  document.body.classList.remove("login-active");
  document.body.classList.add("dashboard-active");
  els.footerUser.textContent = currentUser.email;
  els.footerAccount.hidden = false;

  if (currentTeam && currentProject) {

    els.topbarWorkspace.textContent =
      `${getWorkspaceName(currentTeam)} / ${getProjectName(currentProject)}`;
  } else if (currentTeam) {

    els.topbarWorkspace.textContent =
      getWorkspaceName(currentTeam);
  } else {

    els.topbarWorkspace.textContent = "";
  }
}

els.footerSignOut.addEventListener("click", signOut);


/* ============================================================
   VIEWS
============================================================ */

function showLogin() {

  els.viewLogin.hidden = false;
  els.viewDashboard.hidden = true;
  els.workspaceSidebar.hidden = true;
  els.footerAccount.hidden = true;
  document.body.classList.remove("dashboard-active");
  document.body.classList.add("login-active");

  els.topbarRight.innerHTML = "";
  els.topbarWorkspace.textContent = "";
}


async function showWorkspaces() {

  els.viewLogin.hidden = true;
  els.viewDashboard.hidden = false;
  setWorkspaceSidebarOpen(
    localStorage.getItem(LS_SIDEBAR_COLLAPSED_KEY) !== "true"
  );
  if (!currentProject) {
    renderWorkspaceSelection();
  }

  renderTopbarRight();

  await loadAndRenderWorkspaces();

  if (currentTeam) {
    await loadAndRenderProjects();
  }
}


function showDashboard() {

  els.viewLogin.hidden = true;
  els.viewDashboard.hidden = false;
  if (!currentProject) {
    renderWorkspaceSelection();
  }

  renderTopbarRight();
}


function setWorkspaceSidebarOpen(isOpen) {

  els.workspaceSidebar.hidden = !isOpen;
  localStorage.setItem(LS_SIDEBAR_COLLAPSED_KEY, String(!isOpen));

  const shell =
    els.activeConsole.querySelector(".run-window-shell");

  if (shell) {
    shell.classList.toggle("sidebar-menu-closed", !isOpen);
    shell.querySelectorAll(".workspace-menu-toggle").forEach(button => {
      button.setAttribute("aria-expanded", String(isOpen));
      button.setAttribute(
        "aria-label",
        isOpen ? "Close workspace menu" : "Open workspace menu"
      );
      button.title = isOpen ? "Close workspace menu" : "Open workspace menu";
    });
  }

  els.sidebarToggle.setAttribute("aria-expanded", String(isOpen));
  els.sidebarToggle.setAttribute(
    "aria-label",
    isOpen ? "Close workspace menu" : "Open workspace menu"
  );
  els.sidebarToggle.title = isOpen ? "Close workspace menu" : "Open workspace menu";
}


function mountWorkspaceNavigation(shell) {

  const layout = shell.querySelector(".run-window-layout");
  if (!layout) return;

  layout.prepend(els.workspaceSidebar);
  shell.append(els.sidebarContextMenu);
  shell.classList.toggle("sidebar-menu-closed", els.workspaceSidebar.hidden);
  shell.querySelectorAll(".workspace-menu-toggle").forEach(button => {
    button.setAttribute("aria-expanded", String(!els.workspaceSidebar.hidden));
    button.setAttribute(
      "aria-label",
      els.workspaceSidebar.hidden ? "Open workspace menu" : "Close workspace menu"
    );
  });
}


function renderWorkspaceSelection() {

  els.activeConsole.innerHTML = `
    <div class="run-window-shell">
      <nav class="run-tab-bar" aria-label="Training runs">
        ${renderWorkspaceMenuToggle()}
        <div class="run-tab-pills"></div>
      </nav>
      <div class="run-window-layout">
        <section class="run-window-content console-empty-state" aria-label="Workspace selection">
          <span>PULSE / WORKSPACE</span>
          <p>Select a workspace and project to open its training runs.</p>
        </section>
      </div>
    </div>
  `;

  mountWorkspaceNavigation(els.activeConsole.querySelector(".run-window-shell"));
}


function renderWorkspaceMenuToggle() {
  const isOpen = !els.workspaceSidebar.hidden;
  return `
    <button class="workspace-menu-toggle" type="button" aria-expanded="${isOpen}"
      aria-label="${isOpen ? "Close" : "Open"} workspace menu" title="${isOpen ? "Close" : "Open"} workspace menu">
      <svg viewBox="0 0 20 20" fill="none" aria-hidden="true">
        <path d="M3 5h14M3 10h14M3 15h14" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
      </svg>
    </button>
  `;
}


function signOut() {

  currentUser = null;
  currentTeam = null;
  currentProject = null;
  currentSessions = [];
  currentAccessToken = null;
  currentRefreshToken = null;
  activeSessionId = null;
  autoSelectLiveRun = false;
  commandDrafts = new Map();
  lastRenderedSignature = "";
  teamIntegration = null;
  notifyBaseline = null;
  viewGeneration++;
  refreshGeneration++;

  if (!els.notificationsModal.hidden) {
    closeNotificationsModal();
  }

  workspaceRecords.clear();
  sidebarContextTeam = null;
  els.workspaceList.replaceChildren();
  els.projectList.replaceChildren();
  els.sidebarContextMenu.hidden = true;
  els.sessionList.innerHTML = "";
  els.activeConsole.innerHTML = "";

  sessionStorage.removeItem(
    SS_SESSION_KEY
  );

  els.loginPassword.value = "";

  showLogin();
}


/* ============================================================
   WORKSPACE HELPERS
============================================================ */

function getWorkspaceName(team) {

  // The repo lives on each project now, not on the workspace.
  const name =
    (team.name || "").trim();

  return name || `Workspace ${team.join_code}`;
}

function renderWorkspaceRow(
  team,
  liveCount = 0
) {
  const row =
    document.createElement("button");

  const name = getWorkspaceName(team);
  row.type = "button";
  row.className = "sidebar-nav-item sidebar-workspace-item";
  row.dataset.teamId = String(team.team_id);
  row.title = `${name}${liveCount ? ` · ${liveCount} live runs` : ""}`;
  row.setAttribute(
    "aria-pressed",
    String(currentTeam?.team_id === team.team_id)
  );
  row.innerHTML = `
    <span class="sidebar-item-mark" aria-hidden="true">${escapeHtml(name.slice(0, 1).toUpperCase())}</span>
    <span class="sidebar-item-label">${escapeHtml(name)}</span>
    ${liveCount ? `<span class="sidebar-live-count">${liveCount}</span>` : ""}
  `;
  row.addEventListener("click", () => openWorkspace(team));

  return row;
}

async function renameWorkspace(team, row) {

  const currentName =
    (team.name || "").trim();

  const nextName =
    window.prompt(
      "Workspace name",
      currentName
    );

  if (nextName === null) {

    return; // cancelled
  }

  const trimmed =
    nextName.trim().slice(0, 80);

  if (!trimmed) {

    window.alert("Name cannot be blank.");

    return;
  }

  try {

    await pgPatch(
      "Teams",
      { team_id: `eq.${team.team_id}` },
      { name: trimmed },
      { expectRows: true }
    );

  } catch (error) {

    console.error("Could not rename workspace:", error);

    if (isUnknownColumnError(error)) {

      window.alert(
        "This deployment's Teams table doesn't have a 'name' column -- couldn't save the name."
      );

    } else {

      window.alert(
        `Couldn't rename workspace: ${error.message || error}`
      );
    }

    return;
  }

  team.name = trimmed;

  const nameEl =
    row.querySelector(".sidebar-item-label");

  if (nameEl) {

    nameEl.textContent =
      getWorkspaceName(team);
  }

  const mark = row.querySelector(".sidebar-item-mark");
  if (mark) {
    mark.textContent = getWorkspaceName(team).slice(0, 1).toUpperCase();
  }
  row.title = getWorkspaceName(team);
  els.projectsWorkspace.textContent =
    getWorkspaceName(currentTeam || team);

  // Keep the topbar/crumbs in sync if this is the open workspace.
  if (currentTeam && currentTeam.team_id === team.team_id) {

    currentTeam.name = trimmed;

    renderTopbarRight();
  }
}

const WORKSPACES_EMPTY_TEXT =
  els.workspacesEmpty.textContent;

const PROJECTS_EMPTY_TEXT =
  els.projectsEmpty.textContent;

async function loadAndRenderWorkspaces() {

  const generation = ++viewGeneration;

  if (!currentUser) return;

  els.workspacesEmpty.hidden = true;

  els.workspacesEmpty.textContent =
    WORKSPACES_EMPTY_TEXT;

  try {

    const teams =
      await loadWorkspacesForUser(
        currentUser.id
      );

    if (generation !== viewGeneration || !currentUser) return;

    if (!teams.length) {

      workspaceRecords.clear();
      els.workspaceList.innerHTML = "";

      els.workspacesEmpty.hidden = false;

      return;
    }

    /*
     * Load the recent sessions for every workspace so the
     * workspace picker can show whether anything is currently live.
     *
     * We only need recent sessions because isLive() uses a 5 minute
     * activity window.
     */
    const workspaceStates =
      await Promise.all(
        teams.map(async team => {

          try {

            const projects =
              await loadProjectsForTeam(team);

            if (!projects.length) {

              return {
                team,
                liveCount: 0
              };
            }

            const rows =
              await pgGet("Debug_Sessions", {
                project_id:
                  `in.(${projects.map(project => project.project_id).join(",")})`,

                select:
                  "id,project_id,created_at,telemetry,incidents,agent_logs",

                order:
                  "created_at.desc",

                limit:
                  "50"
              });

            const sessions =
              await Promise.all(
                rows.map(async row => {

                  const telemetry =
                    await decodeEntries(
                      row.telemetry
                    );

                  const incidents =
                    await decodeEntries(
                      row.incidents || []
                    );

                  const agentLogs =
                    await decodeEntries(
                      row.agent_logs || []
                    );

                  return {
                    ...row,
                    telemetry,
                    incidents,
                    agentLogs
                  };
                })
              );

            const liveRuns =
              sessions.filter(
                session =>
                  isLive(session)
              );

            return {
              team,
              liveCount:
                liveRuns.length
            };

          } catch (error) {

            console.warn(
              `Could not load live status for workspace ${team.team_id}:`,
              error
            );

            return {
              team,
              liveCount: 0
            };
          }
        })
      );


    if (generation !== viewGeneration || !currentUser) return;

    workspaceRecords = new Map(
      workspaceStates.map(({ team }) => [String(team.team_id), team])
    );
    const rows = document.createDocumentFragment();

    workspaceStates.forEach(
      ({ team, liveCount }) => {

        rows.appendChild(
          renderWorkspaceRow(
            team,
            liveCount
          )
        );
      }
    );

    els.workspaceList.replaceChildren(rows);

  } catch (error) {

    if (generation !== viewGeneration) return;

    console.error(error);

    els.workspaceList.innerHTML = "";

    els.workspacesEmpty.hidden = false;

    els.workspacesEmpty.textContent =
      "Couldn't load your workspaces. Try refreshing.";
  }
}

async function openWorkspace(team) {

  const workspaceChanged =
    currentTeam?.team_id !== team.team_id;

  currentTeam = team;
  els.projectsWorkspace.textContent =
    getWorkspaceName(team);
  els.projectJoinMsg.textContent = "";

  if (workspaceChanged) {
    currentProject = null;
    currentSessions = [];
    activeSessionId = null;
    lastRenderedSignature = "";
    notifyBaseline = null;
    teamIntegration = null;
    els.sessionList.replaceChildren();
    els.activeConsole.replaceChildren();
    els.projectList.replaceChildren();
    els.projectList.dataset.teamId = String(team.team_id);
    els.projectsEmpty.hidden = false;
    els.projectsEmpty.textContent = "Loading projects…";
    renderStats([]);
  }

  els.workspaceList.querySelectorAll(".sidebar-workspace-item").forEach(item => {
    item.setAttribute("aria-pressed", String(item.dataset.teamId === String(team.team_id)));
  });

  showDashboard();

  await Promise.all([
    loadAndRenderProjects(),
    loadTeamIntegration().catch(error => {
      console.warn("Could not load notification integrations:", error);
    })
  ]);
}


function renderProjectRow(
  project,
  liveCount = 0
) {

  const row =
    document.createElement("button");

  row.type = "button";
  row.className = "sidebar-nav-item sidebar-project-item";
  row.dataset.projectId = String(project.project_id);
  row.title = `${getProjectName(project)}${liveCount ? ` · ${liveCount} live runs` : ""}`;
  row.setAttribute(
    "aria-current",
    currentProject?.project_id === project.project_id ? "page" : "false"
  );

  row.innerHTML = `
    <span class="sidebar-project-mark" aria-hidden="true">
      <svg viewBox="0 0 20 20" fill="none">
        <path d="M3.5 6h5l1.5 1.7h6.5v7.8h-13V6Z" stroke="currentColor" stroke-width="1.3" stroke-linejoin="round"/>
      </svg>
    </span>
    <span class="sidebar-item-label">${escapeHtml(getProjectName(project))}</span>
    ${project.is_secret ? `<span class="sidebar-private-mark" title="Private project">●</span>` : ""}
    ${liveCount ? `<span class="sidebar-live-count">${liveCount}</span>` : ""}
  `;

  row.addEventListener(
    "click",
    () => openProject(project)
  );

  return row;
}


async function loadAndRenderProjects() {

  const team = currentTeam;

  const generation = ++viewGeneration;

  const stale = () =>
    generation !== viewGeneration || team !== currentTeam;

  if (!team) return;

  els.projectsEmpty.hidden = false;
  els.projectsEmpty.textContent = "Loading projects…";

  try {

    const projects =
      await loadProjectsForTeam(team);

    // Switched workspace while this was loading.
    if (stale()) {
      return;
    }

    els.projectsEmpty.textContent =
      PROJECTS_EMPTY_TEXT;

    if (!projects.length) {

      els.projectList.innerHTML = "";

      els.projectsEmpty.hidden = false;

      return;
    }

    els.projectsEmpty.hidden = true;

    const liveCounts = new Map();

    try {

      const sessions =
        await loadSessions(
          projects.map(
            project => project.project_id
          )
        );

      sessions
        .filter(session => isLive(session))
        .forEach(session => {

          liveCounts.set(
            session.project_id,
            (liveCounts.get(session.project_id) || 0) + 1
          );
        });

    } catch (error) {

      console.warn(
        "Could not load live status for projects:",
        error
      );
    }

    projects.sort(
      (a, b) =>
        (liveCounts.get(b.project_id) || 0) -
        (liveCounts.get(a.project_id) || 0)
    );

    if (stale()) {
      return;
    }

    const rows = document.createDocumentFragment();

    projects.forEach(project => {

      rows.appendChild(
        renderProjectRow(
          project,
          liveCounts.get(project.project_id) || 0
        )
      );
    });

    els.projectList.replaceChildren(rows);

  } catch (error) {

    if (stale()) {
      return;
    }

    console.error(error);

    els.projectList.innerHTML = "";

    els.projectsEmpty.hidden = false;

    els.projectsEmpty.textContent =
      "Couldn't load this workspace's projects. Try refreshing.";
  }
}


async function openProject(project) {

  currentProject = project;
  autoSelectLiveRun = true;
  els.projectList.querySelectorAll(".sidebar-project-item").forEach(item => {
    item.setAttribute("aria-current", String(item.dataset.projectId === String(project.project_id) ? "page" : "false"));
  });

  currentSessions = [];

  lastRenderedSignature = "";

  // Fresh project -- don't fire notifications for runs that already
  // existed before this dashboard session opened it.
  notifyBaseline = null;

  activeSessionId = null;

  els.sessionList.innerHTML = "";

  els.activeConsole.innerHTML = "";

  renderStats([]);

  els.workspaceTitle.textContent =
    getProjectName(project);

  els.projectCrumb.textContent =
    `${getWorkspaceName(currentTeam)} / ${project.is_secret ? "SECRET PROJECT" : "PROJECT"}`
      .toUpperCase();

  showDashboard();
  if (window.matchMedia("(max-width: 600px)").matches) {
    setWorkspaceSidebarOpen(false);
  }

  await refreshSessions();
}


/* ============================================================
   NOTIFICATIONS
   Slack / Discord webhooks + browser push, fired when a run goes
   live or logs a new incident.
============================================================ */


/* ============================================================
   NOTIFICATIONS
   Discord / Slack / browser push
============================================================ */

function loadNotificationSettings() {

  try {

    const saved =
      JSON.parse(
        localStorage.getItem(LS_NOTIF_KEY) || "null"
      );

    if (saved && typeof saved === "object") {

      notificationSettings = {
        browserPush: Boolean(saved.browserPush)
      };
    }

  } catch {
    // Ignore corrupted or legacy notification settings.
  }

  renderNotificationsButtonState();
}


function saveNotificationSettingsToStorage() {

  localStorage.setItem(
    LS_NOTIF_KEY,
    JSON.stringify(notificationSettings)
  );
}


/* ============================================================
   TEAM INTEGRATION
============================================================ */

async function loadTeamIntegration() {

  if (!currentTeam) {

    teamIntegration = null;

    renderNotificationsButtonState();

    return;
  }

  /*
   * IMPORTANT:
   * Do not select the Slack webhook here.
   *
   * The webhook is stored in the private
   * team_integration_secrets table and is only
   * accessible to Edge Functions.
   */

  const team = currentTeam;

  const rows =
    await pgGet("team_integrations", {
      team_id: `eq.${team.team_id}`,

      select: [
        "team_id",
        "discord_guild_id",
        "discord_guild_name",
        "discord_channel_id",
        "discord_channel_name",
        "slack_team_id",
        "slack_team_name",
        "slack_channel_id",
        "slack_channel_name",
        "updated_at"
      ].join(",")
    });

  // Another workspace was opened while this loaded.
  if (currentTeam?.team_id !== team.team_id) {
    return;
  }

  teamIntegration =
    rows[0] || null;

  renderNotificationsButtonState();
}


function renderNotificationsButtonState() {

  const configured =
    Boolean(
      teamIntegration?.discord_channel_id ||

      teamIntegration?.slack_team_id ||

      (
        notificationSettings.browserPush &&
        typeof Notification !== "undefined" &&
        Notification.permission === "granted"
      )
    );

  els.notificationsBtn.classList.toggle(
    "is-configured",
    configured
  );

}


/* ============================================================
   BROWSER PUSH
============================================================ */

function browserPushStatusText() {

  if (typeof Notification === "undefined") {

    return "Not supported in this browser.";
  }

  if (Notification.permission === "granted") {

    return "Allowed in this browser.";
  }

  if (Notification.permission === "denied") {

    return "Blocked -- enable notifications for this site in your browser settings.";
  }

  return "You'll be asked to allow notifications when you save.";
}


function sendBrowserNotification(
  title,
  body,
  tag = `pulse-${Date.now()}-${Math.random().toString(36).slice(2)}`
) {

  if (
    typeof Notification === "undefined" ||
    Notification.permission !== "granted"
  ) {
    return;
  }

  try {

    new Notification(title, {
      body,
      icon: PULSE_BOT_ICON_DATA_URI,
      tag
    });

  } catch (error) {

    console.warn(
      "Browser notification failed:",
      error
    );
  }
}


/* ============================================================
   INTEGRATION ROWS
============================================================ */

function renderIntegrationRows() {

  renderDiscordIntegrationRow();

  renderSlackIntegrationRow();
}


/* ============================================================
   DISCORD
============================================================ */

function renderDiscordIntegrationRow() {

  const row =
    document.getElementById(
      "discord-integration-row"
    );

  if (!row) return;


  if (!teamIntegration?.discord_guild_id) {

    row.innerHTML = `
      <button
        type="button"
        class="modal-btn-secondary"
        id="discord-connect-btn"
      >
        Connect Discord
      </button>
    `;

    document
      .getElementById("discord-connect-btn")
      .addEventListener(
        "click",
        () => startOAuthFlow("discord")
      );

    return;
  }


  const currentLabel =
    teamIntegration.discord_channel_name
      ? `# ${teamIntegration.discord_channel_name}`
      : "Choose a channel...";


  row.innerHTML = `
    <span class="integration-status">

      <span
        class="integration-status-dot is-connected"
      ></span>

      <span>
        Connected to
        <strong>
          ${escapeHtml(
            teamIntegration.discord_guild_name ||
            "your server"
          )}
        </strong>
      </span>

    </span>

    <select id="discord-channel-select">

      <option value="">
        ${escapeHtml(currentLabel)}
      </option>

    </select>

    <button
      type="button"
      class="modal-btn-secondary is-small"
      id="discord-disconnect-btn"
    >
      Disconnect
    </button>
  `;


  document
    .getElementById(
      "discord-disconnect-btn"
    )
    .addEventListener(
      "click",
      () => disconnectIntegration("discord")
    );


  const select =
    document.getElementById(
      "discord-channel-select"
    );


  select.addEventListener(
    "focus",
    async () => {

      if (
        select.dataset.loaded === "1"
      ) {
        return;
      }

      select.dataset.loaded = "1";


      try {

        const result =
          await fetchDiscordChannels();

        const channels =
          result?.channels || [];

        const current =
          teamIntegration.discord_channel_id;


        select.innerHTML = `
          <option value="">
            Choose a channel...
          </option>

          ${channels
            .map(channel => `
              <option
                value="${escapeHtml(channel.id)}"
                ${
                  channel.id === current
                    ? "selected"
                    : ""
                }
              >
                # ${escapeHtml(channel.name)}
              </option>
            `)
            .join("")}
        `;

      } catch (error) {

        console.error(
          "Could not load Discord channels:",
          error
        );

        select.innerHTML = `
          <option value="">
            Couldn't load channels
          </option>
        `;

        // opening the list again tries again
        select.dataset.loaded = "";

        els.notifSaveStatus.textContent =
          `Couldn't load Discord channels: ${
            error.message
          }`;

        els.notifSaveStatus.classList.add(
          "is-error"
        );
      }
    }
  );


  select.addEventListener(
    "change",
    async () => {

      const option =
        select.selectedOptions[0];

      if (!option?.value) {
        return;
      }


      try {

        await setDiscordChannel(
          option.value,
          option.textContent
            .trim()
            .replace(/^#\s*/, "")
        );

        await loadTeamIntegration();

        renderDiscordIntegrationRow();

      } catch (error) {

        console.error(
          "Could not set Discord channel:",
          error
        );

        els.notifSaveStatus.textContent =
          `Couldn't save Discord channel: ${
            error.message
          }`;

        els.notifSaveStatus.classList.add(
          "is-error"
        );
      }
    }
  );
}


async function fetchDiscordChannels() {

  if (!currentTeam) {

    return {
      channels: []
    };
  }


  return callEdgeFunction(
    "discord-list-channels",
    {
      method: "GET",

      params: {
        team_id:
          currentTeam.team_id
      }
    }
  );
}


async function setDiscordChannel(
  channelId,
  channelName
) {

  if (!currentTeam) {
    return;
  }


  await pgPatch(
    "team_integrations",
    {
      team_id:
        `eq.${currentTeam.team_id}`
    },
    {
      discord_channel_id:
        channelId,

      discord_channel_name:
        channelName,

      updated_at:
        new Date().toISOString()
    },
    { expectRows: true }
  );
}


/* ============================================================
   SLACK
============================================================ */

function renderSlackIntegrationRow() {

  const row =
    document.getElementById(
      "slack-integration-row"
    );

  if (!row) return;


  /*
   * We deliberately check slack_team_id instead
   * of slack_webhook_url.
   *
   * The webhook is private and never sent
   * to the browser.
   */

  if (!teamIntegration?.slack_team_id) {

    row.innerHTML = `
      <button
        type="button"
        class="modal-btn-secondary"
        id="slack-connect-btn"
      >
        Connect to Slack
      </button>
    `;

    document
      .getElementById("slack-connect-btn")
      .addEventListener(
        "click",
        () => startOAuthFlow("slack")
      );

    return;
  }


  row.innerHTML = `
    <span class="integration-status">

      <span
        class="integration-status-dot is-connected"
      ></span>

      <span>

        Posting to

        <strong>
          #${escapeHtml(
            teamIntegration.slack_channel_name ||
            "unknown"
          )}
        </strong>

        in

        ${escapeHtml(
          teamIntegration.slack_team_name ||
          "your workspace"
        )}

      </span>

    </span>

    <button
      type="button"
      class="modal-btn-secondary is-small"
      id="slack-disconnect-btn"
    >
      Disconnect
    </button>
  `;


  document
    .getElementById(
      "slack-disconnect-btn"
    )
    .addEventListener(
      "click",
      () => disconnectIntegration("slack")
    );
}


/* ============================================================
   OAUTH
============================================================ */

async function startOAuthFlow(
  provider
) {

  if (!currentTeam) {
    return;
  }


  const providerLabel =
    provider === "discord"
      ? "Discord"
      : "Slack";


  els.notifSaveStatus.textContent =
    `Redirecting to ${providerLabel}...`;

  els.notifSaveStatus.classList.remove(
    "is-error"
  );


  try {

    const returnTo =
      window.location.origin +
      window.location.pathname;


    const { state } =
      await callEdgeFunction(
        "create-oauth-state",
        {
          body: {
            team_id:
              currentTeam.team_id,

            provider,

            return_to:
              returnTo
          }
        }
      );


    sessionStorage.setItem(
      SS_PENDING_OAUTH_TEAM_KEY,
      currentTeam.team_id
    );


    const authorizeUrl =
      provider === "discord"

        ? `https://discord.com/oauth2/authorize?${
            new URLSearchParams({

              client_id:
                DISCORD_CLIENT_ID,

              scope:
                "bot",

              permissions:
                DISCORD_BOT_PERMISSIONS,

              redirect_uri:
                `${SUPABASE_URL}/functions/v1/discord-oauth-callback`,

              response_type:
                "code",

              state

            })
          }`

        : `https://slack.com/oauth/v2/authorize?${
            new URLSearchParams({

              client_id:
                SLACK_CLIENT_ID,

              scope:
                "incoming-webhook",

              redirect_uri:
                `${SUPABASE_URL}/functions/v1/slack-oauth-callback`,

              state

            })
          }`;


    window.location.href =
      authorizeUrl;

  } catch (error) {

    console.error(
      `Could not start ${providerLabel} OAuth:`,
      error
    );

    els.notifSaveStatus.textContent =
      `Couldn't start the ${providerLabel} connection: ${
        error.message
      }`;

    els.notifSaveStatus.classList.add(
      "is-error"
    );
  }
}


/* ============================================================
   DISCONNECT
============================================================ */

async function disconnectIntegration(
  provider
) {

  if (!currentTeam) {
    return;
  }


  const clearFields =
    provider === "discord"

      ? {
          discord_guild_id:
            null,

          discord_guild_name:
            null,

          discord_channel_id:
            null,

          discord_channel_name:
            null
        }

      : {
          slack_team_id:
            null,

          slack_team_name:
            null,

          slack_channel_id:
            null,

          slack_channel_name:
            null
        };


  try {

    await pgPatch(
      "team_integrations",

      {
        team_id:
          `eq.${currentTeam.team_id}`
      },

      {
        ...clearFields,

        updated_at:
          new Date().toISOString()
      },
      { expectRows: true }
    );


    /*
     * DO NOT touch slack_webhook_url here.
     *
     * The webhook is in:
     *
     * team_integration_secrets
     *
     * and should only be modified by an Edge Function.
     */


    await loadTeamIntegration();

    renderIntegrationRows();

  } catch (error) {

    console.error(
      "Could not disconnect integration:",
      error
    );

    els.notifSaveStatus.textContent =
      `Couldn't disconnect: ${
        error.message
      }`;

    els.notifSaveStatus.classList.add(
      "is-error"
    );
  }
}


/* ============================================================
   NOTIFICATION MODAL
============================================================ */

async function openNotificationsModal() {

  els.notifBrowserPush.checked =
    notificationSettings.browserPush;


  els.notifPushStatus.textContent =
    browserPushStatusText();


  els.notifSaveStatus.textContent =
    "";

  els.notifSaveStatus.classList.remove(
    "is-error"
  );


  els.notificationsModal.hidden =
    false;


  if (pendingIntegrationNotice) {

    const {
      integration,
      status
    } = pendingIntegrationNotice;


    pendingIntegrationNotice =
      null;


    const providerLabel =
      integration === "discord"
        ? "Discord"
        : "Slack";


    if (status === "connected") {

      els.notifSaveStatus.textContent =
        `${providerLabel} connected.`;

    } else {

      els.notifSaveStatus.textContent =
        `Couldn't connect ${providerLabel}. Please try again.`;

      els.notifSaveStatus.classList.add(
        "is-error"
      );
    }
  }


  try {

    await loadTeamIntegration();

  } catch (error) {

    console.error(
      "Could not load notification integrations:",
      error
    );

    els.notifSaveStatus.textContent =
      `Couldn't load integrations: ${
        error.message
      }`;

    els.notifSaveStatus.classList.add(
      "is-error"
    );
  }


  renderIntegrationRows();
}


function closeNotificationsModal() {

  els.notificationsModal.hidden =
    true;
}


/* ============================================================
   SAVE SETTINGS
============================================================ */

async function handleSaveNotifications() {

  const wantsPush =
    els.notifBrowserPush.checked;


  let pushGranted =
    false;


  if (wantsPush) {

    if (
      typeof Notification ===
      "undefined"
    ) {

      els.notifSaveStatus.textContent =
        "This browser doesn't support push notifications.";

      els.notifSaveStatus.classList.add(
        "is-error"
      );

      return;
    }


    const permission =
      Notification.permission === "granted"

        ? "granted"

        : await Notification.requestPermission();


    pushGranted =
      permission === "granted";


    if (!pushGranted) {

      els.notifSaveStatus.textContent =
        "Browser notifications were blocked. Discord/Slack will still work if connected.";

      els.notifSaveStatus.classList.add(
        "is-error"
      );
    }
  }


  notificationSettings = {
    browserPush:
      wantsPush &&
      pushGranted
  };


  saveNotificationSettingsToStorage();

  renderNotificationsButtonState();


  els.notifPushStatus.textContent =
    browserPushStatusText();


  if (
    !wantsPush ||
    pushGranted
  ) {

    els.notifSaveStatus.textContent =
      "Saved.";

    els.notifSaveStatus.classList.remove(
      "is-error"
    );
  }
}


/* ============================================================
   TEST ALERT
============================================================ */

async function handleTestNotifications() {

  const wantsPush =
    els.notifBrowserPush.checked;


  const hasDiscord =
    Boolean(
      teamIntegration?.discord_channel_id
    );


  const hasSlack =
    Boolean(
      teamIntegration?.slack_team_id
    );


  if (
    !wantsPush &&
    !hasDiscord &&
    !hasSlack
  ) {

    els.notifSaveStatus.textContent =
      "Connect Discord/Slack or enable browser push first.";

    els.notifSaveStatus.classList.add(
      "is-error"
    );

    return;
  }


  els.notifSaveStatus.textContent =
    "Sending test alert...";

  els.notifSaveStatus.classList.remove(
    "is-error"
  );


  try {

    const results =
      await dispatchNotification({
        title:
          "Pulse test alert",

        body:
          `This is a test notification from ${
            currentTeam
              ? getWorkspaceName(currentTeam)
              : "Pulse"
          }.`,

        browserPush:
          wantsPush
      });


    const failures =
      results.filter(
        result => !result.ok
      );


    if (!failures.length) {

      els.notifSaveStatus.textContent =
        "Test alert sent.";

      els.notifSaveStatus.classList.remove(
        "is-error"
      );

    } else {

      els.notifSaveStatus.textContent =
        `Some channels failed: ${
          failures
            .map(
              failure =>
                `${failure.channel}: ${
                  failure.error || "failed"
                }`
            )
            .join("; ")
        }`;

      els.notifSaveStatus.classList.add(
        "is-error"
      );
    }

  } catch (error) {

    console.error(
      "Test notification failed:",
      error
    );

    els.notifSaveStatus.textContent =
      `Test alert failed: ${
        error.message
      }`;

    els.notifSaveStatus.classList.add(
      "is-error"
    );
  }
}


/* ============================================================
   NOTIFICATION DISPATCH
============================================================ */

async function dispatchNotification({
  title,
  body,
  browserPush
}) {

  const jobs = [];


  /* Browser */

  if (browserPush) {

    sendBrowserNotification(
      title,
      body
    );

    jobs.push(
      Promise.resolve({
        channel:
          "Browser",

        // nothing is shown without the browser's permission
        ok:
          typeof Notification !== "undefined" &&
          Notification.permission === "granted"
      })
    );
  }


  /* Discord */

  if (
    teamIntegration?.discord_channel_id &&
    currentTeam
  ) {

    jobs.push(

      callEdgeFunction(
        "send-discord-nonification",
        {
          body: {
            team_id:
              currentTeam.team_id,

            title,

            text: body
          }
        }
      )

        .then(() => ({
          channel:
            "Discord",

          ok:
            true
        }))

        .catch(error => {

          console.error(
            "Discord notification failed:",
            error
          );

          return {
            channel:
              "Discord",

            ok:
              false,

            error:
              error.message
          };
        })
    );
  }


  /* Slack */

  if (
    teamIntegration?.slack_team_id &&
    currentTeam
  ) {

    jobs.push(

      callEdgeFunction(
        "send-slack-notification",
        {
          body: {
            team_id:
              currentTeam.team_id,

            title,

            body
          }
        }
      )

        .then(() => ({
          channel:
            "Slack",

          ok:
            true
        }))

        .catch(error => {

          console.error(
            "Slack notification failed:",
            error
          );

          return {
            channel:
              "Slack",

            ok:
              false,

            error:
              error.message
          };
        })
    );
  }


  return Promise.all(jobs);
}


/* ============================================================
   AUTOMATIC NOTIFICATIONS
============================================================ */

function notifyConfiguredChannels(
  title,
  body
) {

  if (
    !teamIntegration?.discord_channel_id &&
    !teamIntegration?.slack_team_id &&
    !notificationSettings.browserPush
  ) {
    return;
  }


  dispatchNotification({
    title,
    body,
    browserPush:
      notificationSettings.browserPush
  }).catch(error => {

    console.error(
      "Automatic notification failed:",
      error
    );
  });
}


/* ============================================================
   NOTIFICATION STATE
============================================================ */

function computeNotifyState(
  sessions
) {

  const liveIds =
    new Set(
      sessions
        .filter(isLive)
        .map(
          session =>
            session.id
        )
    );


  const incidentKeys =
    new Set();


  sessions.forEach(
    session => {

      (
        session.incidents ||
        []
      ).forEach(
        incident => {

          incidentKeys.add(
            `${session.id}:${
              incident.t || 0
            }:${
              incident.kind ||
              "event"
            }`
          );
        }
      );
    }
  );


  return {
    liveIds,
    incidentKeys
  };
}


function updateNotifyBaseline(
  sessions
) {

  const nextState =
    computeNotifyState(
      sessions
    );


  if (!notifyBaseline) {

    notifyBaseline =
      nextState;

    return;
  }


  const workspaceName =
    currentTeam
      ? getWorkspaceName(
          currentTeam
        )
      : "your workspace";


  const sessionsById =
    new Map(
      sessions.map(
        session => [
          session.id,
          session
        ]
      )
    );


  nextState.liveIds.forEach(
    id => {

      if (
        notifyBaseline.liveIds.has(
          id
        )
      ) {
        return;
      }


      const session =
        sessionsById.get(id);


      const who =
        session?.username ||
        "A run";


      notifyConfiguredChannels(
        "Pulse: run is live",
        `${who}'s run in ${workspaceName} just went live.`
      );
    }
  );


  nextState.incidentKeys.forEach(
    key => {

      if (
        notifyBaseline.incidentKeys.has(
          key
        )
      ) {
        return;
      }


      const [
        sessionId
      ] = key.split(":");


      const session =
        sessionsById.get(
          sessionId
        );


      const who =
        session?.username ||
        "A run";


      const incident =
        (
          session?.incidents ||
          []
        ).find(
          candidate =>
            `${
              sessionId
            }:${
              candidate.t || 0
            }:${
              candidate.kind ||
              "event"
            }`
            === key
        );


      notifyConfiguredChannels(
        "Pulse: new incident",
        `${incident?.kind || "Incident"} on ${who}'s run in ${workspaceName}${
          incident?.summary
            ? `: ${incident.summary}`
            : "."
        }`
      );
    }
  );


  notifyBaseline =
    nextState;
}

/* ============================================================
   STATS
============================================================ */

function renderStats(sessions) {

  const liveCount =
    sessions.filter(isLive).length;

  const incidentCount =
    sessions.reduce(
      (count, session) =>
        count +
        (session.incidents?.length || 0),
      0
    );

  if (els.runCount) {
    els.runCount.textContent = String(sessions.length);
  }
}


/* ============================================================
   INCIDENT
============================================================ */
function renderIncident(incident) {
  const neutralKinds = new Set([
    "fix_applied",
    "revert"
  ]);

  const kind = incident.kind || "event";
  const neutral = neutralKinds.has(kind);

  const time = incident.t
    ? new Date(incident.t * 1000).toLocaleString(
        undefined,
        {
          month: "short",
          day: "numeric",
          hour: "2-digit",
          minute: "2-digit"
        }
      )
    : "—";

  return `
    <li class="incident">

      <div class="incident-main">

        <span class="incident-time">
          ${escapeHtml(time)}
        </span>

        <span>
          <span class="
            incident-kind
            ${neutral ? "is-neutral" : ""}
          ">
            ${escapeHtml(kind)}
          </span>

          ${escapeHtml(incident.summary || "")}
        </span>

      </div>

      <div class="incident-distribution">
        <div class="incident-distribution-inner">

          <!-- Put your existing distribution/details content here -->

        </div>
      </div>

    </li>
  `;
}
/* ============================================================
   ML ANTI-PATTERNS (pulled from telemetry)
============================================================ */

const REGRESSION_LOSSES = new Set([
  "mse", "mae", "mean_squared_error", "mean_absolute_error",
  "msle", "mean_squared_logarithmic_error", "huber", "logcosh"
]);

const ACCURACY_METRICS = new Set([
  "accuracy", "acc", "categorical_accuracy", "binary_accuracy",
  "sparse_categorical_accuracy", "top_k_categorical_accuracy"
]);

function detectMetricLossMismatch(session) {
  // Check if telemetry has a regression loss + accuracy metric combo
  if (!session.telemetry || session.telemetry.length === 0) {
    return null;
  }

  const allKeys = new Set();
  session.telemetry.forEach(entry => {
    if (entry && typeof entry === "object") {
      Object.keys(entry).forEach(k => allKeys.add(k.toLowerCase()));
    }
  });

  // Find any regression loss name
  let hasRegression = false;
  for (const key of allKeys) {
    if (REGRESSION_LOSSES.has(key)) {
      hasRegression = true;
      break;
    }
  }

  // Find any accuracy metric name
  let hasAccuracy = false;
  for (const key of allKeys) {
    if (ACCURACY_METRICS.has(key)) {
      hasAccuracy = true;
      break;
    }
  }

  return (hasRegression && hasAccuracy) ? true : null;
}

function extractLossCurve(session) {
  // Extract loss history from telemetry
  if (!session.telemetry || session.telemetry.length === 0) {
    return { steps: [], losses: [] };
  }

  const lossNames = ["loss", "train_loss", "loss_value", "current_loss"];
  const steps = [];
  const losses = [];

  session.telemetry.forEach((entry, idx) => {
    if (!entry) return;

    // Use step if available, otherwise use index
    const step = entry.step !== undefined ? entry.step : idx;
    steps.push(step);

    // Find first matching loss name
    let lossVal = null;
    for (const name of lossNames) {
      if (entry[name] !== undefined && entry[name] !== null) {
        lossVal = entry[name];
        break;
      }
    }
    losses.push(lossVal);
  });

  return { steps, losses };
}

const MAX_LOSS_CHART_POINTS = 300;

function downsamplePoints(points, maxPoints) {

  if (points.length <= maxPoints) {
    return points;
  }

  const stride = Math.ceil(points.length / maxPoints);

  return points.filter((_, idx) => idx % stride === 0);
}

function renderLossCurve(session) {
  const { steps, losses } = extractLossCurve(session);

  if (losses.length === 0 || losses.every(v => v === null)) {
    return "";
  }

  // Filter out null values for charting
  const validPoints = downsamplePoints(
    losses
      .map((loss, idx) => ({ step: steps[idx], loss }))
      .filter(p => p.loss !== null && p.loss !== undefined && isFinite(p.loss)),
    MAX_LOSS_CHART_POINTS
  );

  if (validPoints.length === 0) {
    return "";
  }

  const minLoss = Math.min(...validPoints.map(p => p.loss));
  const maxLoss = Math.max(...validPoints.map(p => p.loss));
  const padding = (maxLoss - minLoss) * 0.1 || 1;

  const chartPayload = {
    steps: validPoints.map(p => p.step),
    losses: validPoints.map(p => p.loss),
    min: minLoss - padding,
    max: maxLoss + padding
  };

  return `
    <div class="loss-curve-container">
      <p class="loss-curve-label">Loss Curve</p>
      <div class="loss-curve-chart">
        <canvas
          data-loss-chart="1"
          data-loss='${escapeHtml(JSON.stringify(chartPayload))}'
        ></canvas>
      </div>
    </div>
  `;
}


/* ============================================================
   LOSS CHART (Chart.js instantiation)
============================================================ */

function destroyLossChart(scopeEl) {

  if (!scopeEl) return;

  const canvas =
    scopeEl.querySelector("canvas[data-loss-chart]");

  if (!canvas) return;

  const chart =
    typeof Chart !== "undefined" &&
    Chart.getChart
      ? Chart.getChart(canvas)
      : null;

  if (chart) {
    chart.destroy();
  }
}


function initLossChart(scopeEl) {

  if (typeof Chart === "undefined") return;

  const canvas =
    scopeEl.querySelector("canvas[data-loss-chart]");

  if (!canvas || canvas.dataset.chartInit === "1") {
    return;
  }

  let payload;

  try {
    payload = JSON.parse(canvas.dataset.loss);
  } catch {
    return;
  }

  new Chart(canvas, {
    type: "line",
    data: {
      labels: payload.steps,
      datasets: [{
        label: "Loss",
        data: payload.losses,
        borderColor: "#ed8a32",
        backgroundColor: "rgba(237, 138, 50, 0.08)",
        borderWidth: 2,
        fill: true,
        tension: 0.1,
        pointRadius: 0,
        pointHoverRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }
      },
      scales: {
        y: {
          min: payload.min,
          max: payload.max,
          grid: { color: "rgba(29, 29, 31, 0.05)" }
        },
        x: {
          grid: { display: false }
        }
      }
    }
  });

  canvas.dataset.chartInit = "1";
}

/* ============================================================
   SESSION
============================================================ */

function renderRunWorkspace(session) {

  const logs = (session.agentLogs || []).filter(entry => entry && typeof entry === "object");
  const questions = (session.commands || []).filter(command => questionOf(command));
  const commands = (session.commands || []).filter(command => !isRunnerRow(command));
  const openQuestion = questions.filter(command => command.status === "processing").at(-1);
  const step = findMetric(session, ["step", "global_step", "epoch"]);
  const live = isLive(session);
  const telemetry = session.telemetry || [];
  // the latest snapshot of the run's state (other entries: the environment, ...)
  const latest = [...telemetry].reverse().find(entry => Array.isArray(entry?.findings)) ||
    latestTelemetry(session);
  const metricKeys = new Set();
  telemetry.forEach(entry => {
    if (entry?.type === "env_info" || entry?.script || entry?.gpu_name || entry?.python_version || entry?.python) {
      return;
    }
    Object.entries(entry || {}).forEach(([key, value]) => {
      if (!["t", "step", "gaps", "gpu_count", "finished"].includes(key) &&
          typeof value === "number" && Number.isFinite(value)) {
        metricKeys.add(key);
      }
    });
  });
  const metricRows = [...metricKeys].sort((left, right) => {
    const leftLoss = /loss/i.test(left) ? 0 : 1;
    const rightLoss = /loss/i.test(right) ? 0 : 1;
    return leftLoss - rightLoss || left.localeCompare(right);
  }).map(name => {
    const values = telemetry
      .filter(entry => typeof entry?.[name] === "number" && Number.isFinite(entry[name]))
      .slice(-48)
      .map(entry => entry[name]);
    return `
      <div class="run-metric-row">
        <span class="run-metric-name" title="${escapeHtml(name)}">${escapeHtml(name)}</span>
        <strong data-live-metric="${escapeHtml(name)}">${escapeHtml(metricValue(values.at(-1)))}</strong>
        <span class="run-metric-spark" aria-label="Recent ${escapeHtml(name)} values">${escapeHtml(metricSparkline(values))}</span>
      </div>
    `;
  }).join("");
  const latestRunStatus = [...telemetry]
    .reverse()
    .find(entry => entry.status)?.status || [...(session.incidents || [])]
      .reverse()
      .find(incident => String(incident.kind || "").startsWith("run_"))?.status;
  // What the runner last said, unless it has gone quiet since: a run last seen "live" whose
  // runner stopped reporting is not live any more.
  const runStatus = live
    ? (latestRunStatus || "live")
    : (["live", "stalled", "paused"].includes(String(latestRunStatus || "").toLowerCase())
        ? "stopped" : (latestRunStatus || "stopped"));
  const stepRate = telemetryStepRate(telemetry);
  const startedAt = session.created_at
    ? Date.parse(session.created_at) / 1000
    : Number(telemetry[0]?.t || 0);
  const elapsed = live && startedAt
    ? fmtDuration(Math.max(Number(session.uptime_seconds || 0), Math.floor(Date.now() / 1000 - startedAt)))
    : fmtDuration(session.uptime_seconds || 0);
  const findings = Array.isArray(latest.findings) ? latest.findings : [];
  const tensors = Object.entries(latest.tensors || {}).slice(0, 5);
  const experiments = (session.incidents || [])
    .filter(incident => incident.kind === "experiment" && incident.experiment)
    .slice(-3)
    .reverse();
  const hasRunStats = Number(step) > 0 || metricKeys.size > 0 || findings.length > 0 ||
    tensors.length > 0 || experiments.length > 0 || Number(session.uptime_seconds || 0) > 0;
  const canSend = Boolean(currentUser && (
    currentUser.id === session.user_id ||
    (currentTeam?.admin_ids || []).includes(currentUser.id)
  ));

  // A prompt sent from here is shown with its state and what the runner answered; the agent
  // turn it caused is the same exchange, so it is not shown twice.
  const commandTimes = commands.map(command => ({
    text: command.command,
    from: command.created_at ? Date.parse(command.created_at) / 1000 - 5 : 0,
    to: command.completed_at ? Date.parse(command.completed_at) / 1000 + 120 : Infinity
  }));
  const isCommandTurn = entry => commandTimes.some(c =>
    c.text === entry.question && Number(entry.t || 0) >= c.from && Number(entry.t || 0) <= c.to);
  const commandState = command => {
    const status = command.status || "pending";
    if (status === "pending") return live ? "waiting" : "waiting · run not connected";
    if (status === "processing") return "running";
    if (status === "completed") return "done";
    return status;
  };

  const entries = logs.filter(entry => !isCommandTurn(entry)).map(entry => {
    // turns Pulse started itself (a crash, a finding, an audit) are not the user's words
    const automatic = entry.who === "pulse" || (!entry.who && entry.traceback_signature);
    const question = automatic
      ? `<details class="console-message is-pulse-prompt"><summary>${escapeHtml(String(entry.question || "").split("\n")[0].slice(0, 140))}</summary><pre>${escapeHtml(entry.question || "")}</pre></details>`
      : `<pre class="console-message is-user">${escapeHtml(entry.question || "")}</pre>`;
    return {
      at: Number(entry.t || 0),
      html: `
      <article class="console-exchange">
        <div class="console-speaker">${automatic ? "PULSE · ON ITS OWN" : entry.who === "dashboard" ? "YOU · FROM HERE" : "YOU"} <time>${fmtRelativeTime(entry.t || 0)}</time></div>
        ${question}
        <div class="console-speaker">PULSE</div>
        <pre class="console-message is-agent">${renderCliText(entry.answer || "")}</pre>
      </article>
    `
    };
  });

  commands.forEach(command => {
    entries.push({
      at: command.created_at ? Date.parse(command.created_at) / 1000 : 0,
      html: `
        <article class="console-exchange">
          <div class="console-speaker">YOU
            <time>${fmtRelativeTime(command.created_at ? Date.parse(command.created_at) / 1000 : 0)}</time>
            <span class="command-state is-${escapeHtml(command.status || "pending")}">${escapeHtml(commandState(command))}</span>
          </div>
          <pre class="console-message is-user is-command">${renderCliText(command.command || "")}</pre>
          ${command.result ? `<div class="console-speaker">PULSE</div><pre class="console-message is-pulse-output">${renderCliText(command.result)}</pre>` : ""}
        </article>
      `
    });
  });

  questions.filter(command => command !== openQuestion).forEach(command => {
    const asked = questionOf(command);
    const answer = String(command.result || "");
    const answerText = /^y(?:es)?$/i.test(answer.trim()) ? "Yes"
      : /^n(?:o)?$/i.test(answer.trim()) ? "No"
        : asked.options && /^#\d+/.test(answer)
          ? asked.options[Number(answer.slice(1).split(" ")[0])] ?? answer
          : answer || "(Enter)";
    const outcome = command.status === "completed"
      ? (String(command.result || "").startsWith("Answered at the machine")
          ? command.result
          : `Answered here: ${answerText}`)
      : command.result || "No longer asked.";
    entries.push({
      at: command.created_at ? Date.parse(command.created_at) / 1000 : 0,
      html: `
        <article class="console-exchange">
          <div class="console-speaker">PULSE ASKED <time>${fmtRelativeTime(command.created_at ? Date.parse(command.created_at) / 1000 : 0)}</time></div>
          <pre class="console-message is-pulse-prompt">${renderCliText(asked.label)}</pre>
          <pre class="console-message is-pulse-output">${renderCliText(outcome)}</pre>
        </article>
      `
    });
  });

  entries.sort((a, b) => a.at - b.at);
  if (!entries.length) {
    entries.push({ html: `<p class="console-empty">No conversation yet. Send a prompt to this run.</p>` });
  }

  const incidents = session.incidents || [];
  const transcript = entries.map(entry => entry.html).join("");
  const incidentRows = incidents.slice(0, 3).map(incident => `
    <div class="run-side-incident">
      <span>${escapeHtml(incident.kind || "event")}</span>
      <p>${escapeHtml(incident.summary || "")}</p>
    </div>
  `).join("");
  const statsPane = hasRunStats ? `
    <aside class="run-side-panel" aria-label="Run telemetry">
      <div class="run-side-heading">RUN STATE</div>
      <div class="run-side-stat"><span>STEP</span><strong data-live-step>${escapeHtml(metricValue(step))}</strong></div>
      <div class="run-side-stat"><span>RATE</span><strong>${escapeHtml(stepRate || "—")}</strong></div>
      <div class="run-side-stat"><span>ELAPSED</span><strong>${escapeHtml(elapsed)}</strong></div>
      ${Number(latest.gaps) > 0 ? `<div class="run-side-stat"><span>DROPPED</span><strong>${escapeHtml(String(latest.gaps))} samples</strong></div>` : ""}
      <div class="run-side-heading run-values-heading">VALUES</div>
      ${metricRows || `<span class="run-side-quiet">no values yet</span>`}
      <div class="run-side-rule"></div>
      <div class="run-side-heading">FINDINGS <b>${findings.length}</b></div>
      ${findings.length ? findings.slice(0, 5).map(finding => `
        <div class="run-side-finding is-${escapeHtml(String(finding.severity || "info").toLowerCase())}">
          <span>${escapeHtml(finding.severity || "info")}</span>
          <p>${escapeHtml(finding.message || "")}</p>
        </div>
      `).join("") : `<span class="run-side-quiet">${latest.step ? "Nothing the checks can see" : "Waiting for the first steps"}</span>`}
      ${experiments.length ? `
        <div class="run-side-rule"></div>
        <div class="run-side-heading">PROXY EXPERIMENTS <b>${experiments.length}</b></div>
        ${experiments.map(incident => {
          const result = incident.experiment;
          const reproduction = result.reproduction || {};
          const confidence = Number(reproduction.confidence || 0);
          const branches = (result.branches || []).filter(branch => branch.name !== "control");
          return `
            <div class="run-side-experiment">
              <span>${escapeHtml(String(result.status || "unknown").toUpperCase())} · ${escapeHtml(result.id || "")}</span>
              <p>${escapeHtml(result.conclusion || incident.summary || "")}</p>
              <small>Reproduction: ${escapeHtml(reproduction.passed ? "passed" : "not established")} · ${escapeHtml(`${Math.round(confidence * 100)}% confidence`)}</small>
              ${branches.slice(0, 3).map(branch => `
                <small>${escapeHtml(branch.name)}: ${escapeHtml(`${Math.round(Number(branch.success_rate || 0) * 100)}% healthy`)} · ${escapeHtml(branch.hypothesis || "")}</small>
                <small>Mean objective: ${escapeHtml(metricValue(branch.mean_metric))} · ${escapeHtml(`${Number(branch.resource_seconds || 0).toFixed(1)}s compute`)}${branch.validated ? " · source-scale validated" : ""}</small>
              `).join("")}
            </div>
          `;
        }).join("")}
      ` : ""}
      ${tensors.length ? `
        <div class="run-side-rule"></div>
        <div class="run-side-heading">TENSORS</div>
        ${tensors.map(([name, meta]) => {
          const shape = Array.isArray(meta?.shape) ? meta.shape.join("×") : "";
          return `<div class="run-side-tensor"><span>${escapeHtml(name)}</span><b>${escapeHtml(`${shape} ${meta?.dtype || ""}`.trim())}</b></div>`;
        }).join("")}
      ` : ""}
      <div class="run-side-rule"></div>
      <div class="run-side-heading">MACHINE</div>
      <div class="run-side-stat"><span>GPU</span><strong>${escapeHtml(session.env?.gpu_name || "Not reported")}</strong></div>
      <div class="run-side-stat"><span>PYTHON</span><strong>${escapeHtml(session.env?.python_version || "n/a")}</strong></div>
      <div class="run-side-stat"><span>UPTIME</span><strong>${escapeHtml(fmtDuration(session.uptime_seconds || 0))}</strong></div>
      <div class="run-side-rule"></div>
      <div class="run-side-heading">INCIDENTS <b>${incidents.length}</b></div>
      ${incidentRows || `<span class="run-side-quiet">No incidents</span>`}
    </aside>
  ` : "";

  return `
    <section class="run-workspace ${hasRunStats ? "has-run-stats" : ""}" aria-label="Run console">
      <header class="run-window-bar">
        <span class="window-lights" aria-hidden="true"><i></i><i></i><i></i></span>
        <span class="run-window-title">PULSE / DEBUG <b>${escapeHtml(runLabel(session))}</b></span>
        <span class="run-window-live is-${escapeHtml(String(runStatus).toLowerCase())}">${escapeHtml(runStatus.toUpperCase())}</span>
      </header>
      <div class="run-window-body ${hasRunStats ? "has-run-stats" : "is-conversation-only"}">
        ${statsPane}
        <section class="run-transcript" aria-label="Agent transcript">${transcript}<div class="run-live" aria-live="polite"></div></section>
      </div>
      ${openQuestion ? renderOpenQuestion(openQuestion, canSend) : ""}
      <form class="command-composer" data-run-id="${escapeHtml(session.id)}">
        <label class="sr-only" for="command-${escapeHtml(session.id)}">Send a prompt to this run</label>
        <div class="command-input-wrap">
        <div class="command-suggestions" role="listbox" aria-label="Pulse commands" hidden></div>
        <textarea class="command-field" id="command-${escapeHtml(session.id)}" name="command" rows="2" maxlength="8000"
          placeholder="${openQuestion ? "Answer the question above before sending another prompt" : "Ask about this run or enter a /command…"}"
          ${canSend && !openQuestion ? "" : "disabled"} required></textarea>
        </div>
        <div class="command-composer-foot">
          <span class="command-status" role="status">${!canSend ? "Only the run owner or workspace admins can send commands"
            : openQuestion ? "Respond to the pending question above to continue"
            : live ? "Enter to send · runs on the machine Pulse is watching it from"
            : "This run's Pulse is not connected: a prompt waits until it is"}</span>
          <button type="submit" title="Send command" ${canSend && !openQuestion ? "" : "disabled"}>Send <span aria-hidden="true">↗</span></button>
        </div>
      </form>
    </section>
  `;
}


function renderOpenQuestion(command, canSend) {
  const asked = questionOf(command);
  const id = escapeHtml(command.id);
  const titleId = `question-title-${id}`;
  const promptId = `question-prompt-${id}`;
  let choices;
  if (asked.options) {
    choices = asked.options.map((option, index) =>
      `<button type="button" class="question-answer" data-question-id="${id}" data-answer="#${index} ${escapeHtml(option)}" ${canSend ? "" : "disabled"}>${escapeHtml(option)}</button>`
    ).join("");
  } else if (isYesNo(asked.label)) {
    choices = `
      <button type="button" class="question-answer is-primary" data-question-id="${id}" data-answer="y" ${canSend ? "" : "disabled"}>Yes</button>
      <button type="button" class="question-answer" data-question-id="${id}" data-answer="n" ${canSend ? "" : "disabled"}>No</button>`;
  } else {
    choices = `<form class="question-text" data-question-id="${id}">
      <input type="text" name="answer" placeholder="Your answer" ${canSend ? "" : "disabled"}>
      <button type="submit" ${canSend ? "" : "disabled"}>Answer</button>
    </form>`;
  }
  return `
    <div class="${canSend ? "run-question-backdrop" : "run-question-readonly"}">
      <section class="run-question" role="${canSend ? "alertdialog" : "region"}" ${canSend ? 'aria-modal="true"' : ""}
        aria-labelledby="${titleId}" aria-describedby="${promptId}" ${canSend ? 'tabindex="-1"' : ""}>
        <div class="approval-eyebrow"><span class="approval-indicator"></span> PULSE NEEDS YOUR APPROVAL</div>
        <h2 id="${titleId}">Approval request</h2>
        <p class="question-prompt" id="${promptId}">${escapeHtml(displayQuestionLabel(asked.label))}</p>
        ${asked.context ? `<details class="approval-context"><summary>Recent context</summary><pre>${escapeHtml(asked.context)}</pre></details>` : ""}
        ${asked.detail ? `<details class="question-detail"><summary>Review the change</summary><pre>${escapeHtml(asked.detail)}</pre></details>` : ""}
        <div class="question-choices">${choices}</div>
        <p class="question-status" role="status">${canSend
          ? isYesNo(asked.label) ? "Choose Yes or No, or press Y / N."
            : "Your answer will be sent to Pulse on the machine."
          : "Only the run owner or workspace admins can answer."}</p>
      </section>
    </div>
  `;
}


async function sendQuestionAnswer(commandId, answer) {
  const panel = els.activeConsole.querySelector(".run-question");
  const status = panel?.querySelector(".question-status");
  panel?.querySelectorAll("button, input").forEach(element => { element.disabled = true; });
  if (status) status.textContent = "Sending…";
  try {
    await answerQuestion(commandId, answer);
    currentSessions.forEach(session => (session.commands || []).forEach(command => {
      if (command.id === commandId) {
        command.status = "completed";
        command.result = answer;
      }
    }));
    lastRenderedSignature = "";
    renderSessions();
  } catch (error) {
    console.warn("Could not answer:", error);
    if (status) status.textContent = /admin/.test(error.message)
      ? "Already answered (at the machine), or you can't answer this run's questions."
      : "Could not send the answer. Try again.";
    panel?.querySelectorAll("button, input").forEach(element => { element.disabled = false; });
    if (/admin/.test(error.message)) refreshSessions();
  }
}


function metricSparkline(values) {

  const glyphs = "▁▂▃▄▅▆▇█";
  if (!values.length) return "";
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min;
  return values.map(value => {
    const index = span === 0 ? 3 : Math.min(7, Math.floor(((value - min) / span) * 7));
    return glyphs[index];
  }).join("");
}


function telemetryStepRate(telemetry) {

  const samples = telemetry.filter(entry =>
    Number.isFinite(Number(entry?.step)) && Number.isFinite(Number(entry?.t))
  ).slice(-2);
  if (samples.length < 2) return "—";
  const seconds = Number(samples[1].t) - Number(samples[0].t);
  const steps = Number(samples[1].step) - Number(samples[0].step);
  if (seconds <= 0 || steps < 0) return "—";
  const rate = steps / seconds;
  return `${rate >= 1 ? rate.toFixed(1) : (rate * 60).toFixed(1)}${rate >= 1 ? "/s" : "/min"}`;
}


function renderSession(session) {

  const live =
    isLive(session);

  const hasIncident =
    (session.incidents || []).length > 0;

  const hasMetricLossMismatch =
    detectMetricLossMismatch(session);

  const uptime =
    session.uptime_seconds || 0;

  const downtime =
    session.downtime_seconds || 0;

  const total =
    uptime + downtime;

  const uptimePct =
    total > 0
      ? Math.round(
          (uptime / total) * 100
        )
      : 100;

  const downtimePct =
    total > 0
      ? 100 - uptimePct
      : 0;

  const commit =
    session.git_commit_sha &&
    session.git_commit_sha !== "unknown"
      ? session.git_commit_sha.slice(0, 10)
      : "no commit";

  const gpu =
    session.env?.gpu_name
      ? `${
          session.env.gpu_count || 1
        }× ${session.env.gpu_name}`
      : "No GPU info";

  const who =
    session.username ||
    "unknown user";

  const loss =
    findMetric(
      session,
      [
        "loss",
        "train_loss",
        "loss_value",
        "current_loss"
      ]
    );

  const grad =
    findMetric(
      session,
      [
        "grad_norm",
        "gradient_norm",
        "grad"
      ]
    );

  const learningRate =
    findMetric(
      session,
      [
        "learning_rate",
        "lr",
        "learningRate"
      ]
    );

  const epoch =
    findMetric(
      session,
      [
        "epoch",
        "step",
        "global_step"
      ]
    );

  const statusClass =
    live
      ? "is-live"
      : hasIncident
        ? "has-incident"
        : "";

  const wrapper =
    document.createElement("div");

  wrapper.className = "session";

  wrapper.dataset.sessionId =
    session.id;

  wrapper.dataset.live =
    live ? "1" : "0";

  wrapper.dataset.incident =
    hasIncident ? "1" : "0";

  wrapper.dataset.renderSignature =
    JSON.stringify([
      session.created_at,
      session.git_commit_sha,
      session.uptime_seconds,
      session.downtime_seconds,
      session.incidents?.length || 0,
      session.telemetry?.length || 0,
      session.agentLogs?.length || 0,
      session.commands?.length || 0,
      session.commands?.at(-1)?.status || "",
      session.errorTracebacks?.length || 0,
      live
    ]);


  const detailGrid = `

    <div class="detail-grid">

      <div>
        <p class="detail-item-label">
          Started by
        </p>
        <p class="detail-item-value">
          ${escapeHtml(who)}
        </p>
      </div>

      <div>
        <p class="detail-item-label">
          Commit
        </p>
        <p class="detail-item-value">
          ${escapeHtml(commit)}
        </p>
      </div>

      <div>
        <p class="detail-item-label">
          GPU
        </p>
        <p class="detail-item-value">
          ${escapeHtml(gpu)}
        </p>
      </div>

      <div>
        <p class="detail-item-label">
          CUDA
        </p>
        <p class="detail-item-value">
          ${escapeHtml(
            session.env?.cuda_version ||
            "n/a"
          )}
        </p>
      </div>

      <div>
        <p class="detail-item-label">
          Python
        </p>
        <p class="detail-item-value">
          ${escapeHtml(
            session.env?.python_version ||
            "n/a"
          )}
        </p>
      </div>

      <div>
        <p class="detail-item-label">
          Uptime
        </p>
        <p class="detail-item-value">
          ${fmtDuration(uptime)}
        </p>
      </div>

      <div>
        <p class="detail-item-label">
          Downtime
        </p>
        <p class="detail-item-value">
          ${fmtDuration(downtime)}
        </p>
      </div>

      <div>
        <p class="detail-item-label">
          Agent turns
        </p>
        <p class="detail-item-value">
          ${(session.agentLogs || []).length}
        </p>
      </div>

    </div>
  `;


  const incidents =
    session.incidents || [];


  wrapper.innerHTML = `

    <button
      class="session-row"
      type="button"
      aria-expanded="false"
    >

      <span
        class="status-dot ${statusClass}"
        aria-hidden="true"
      ></span>


      <span class="session-main">

        <span class="session-title">

          ${escapeHtml(who)}

          <span class="session-commit">
            ${escapeHtml(commit)}
          </span>

        </span>

        <span class="session-meta">

          <span>
            ${escapeHtml(gpu)}
          </span>

          <span>
            ${fmtRelativeTime(
              session.created_at
                ? Date.parse(
                    session.created_at
                  ) / 1000
                : null
            )}
          </span>

          ${
            hasIncident
              ? `
                <span>
                  ${incidents.length}
                  incident${
                    incidents.length === 1
                      ? ""
                      : "s"
                  }
                </span>
              `
              : ""
          }

        </span>

      </span>


      <span class="run-metric">

        <span class="run-metric-label">
          Loss
        </span>

        <span class="run-metric-value">
          ${metricValue(loss)}
        </span>

      </span>


      <span class="run-metric">

        <span class="run-metric-label">
          Grad norm
        </span>

        <span class="run-metric-value">
          ${metricValue(grad)}
        </span>

      </span>


      <span class="run-metric">

        <span class="run-metric-label">
          Learning rate
        </span>

        <span class="run-metric-value">
          ${metricValue(learningRate)}
        </span>

      </span>


      <span class="run-health">

        <span class="run-metric-label">
          Runtime
        </span>

        <span class="uptime-bar">

          <span
            class="uptime-bar-fill"
            style="width:${uptimePct}%"
          ></span>

          <span
            class="downtime-bar-fill"
            style="width:${downtimePct}%"
          ></span>

        </span>

        <span class="run-health-label">
          ${uptimePct}% uptime
        </span>

      </span>


      <span class="session-caret">
        ›
      </span>

    </button>


    <div class="session-detail">

      <div class="uptime-summary">

        <span>
          <strong>${uptimePct}%</strong>
          uptime
        </span>

        <span class="is-downtime">
          <strong>${downtimePct}%</strong>
          downtime
        </span>

        <span class="tracked-time">
          ${fmtTrackedTime(total)}
          tracked
        </span>

      </div>

      ${renderRunWorkspace(session)}

      ${detailGrid}

      ${
        hasMetricLossMismatch
          ? `
            <div class="warning-banner">
              <span class="warning-icon">⚠</span>
              <span class="warning-text">
                <strong>ML Setup Issue Detected:</strong> This run has a regression loss (MSE/MAE) 
                but tracks an accuracy metric, which is fundamentally incompatible. 
                Accuracy is an exact-match classifier metric; it will remain stuck near random 
                chance for a continuous target. Either fix the loss/metric mismatch or remove the 
                accuracy metric entirely.
              </span>
            </div>
          `
          : ""
      }

      ${renderLossCurve(session)}

      <p class="incidents-title">
        Incidents (${incidents.length})
      </p>

      ${
        incidents.length
          ? `
            <ul class="incident-list">
              ${incidents
                .map(renderIncident)
                .join("")}
            </ul>
          `
          : `
            <p class="no-incidents">
              Nothing logged for this run.
            </p>
          `
      }

    </div>
  `;


  const row =
    wrapper.querySelector(
      ".session-row"
    );


  row.addEventListener(
    "click",
    () => {

      const open =
        wrapper.classList.toggle(
          "is-open"
        );

      row.setAttribute(
        "aria-expanded",
        String(open)
      );

      if (open) {
        initLossChart(wrapper);
      }
    }
  );


  return wrapper;
}


/* ============================================================
   SESSION LIST
============================================================ */

function renderLegacySessions() {

  const filtered =
    currentSessions.filter(
      session => {

        if (
          currentFilter === "live"
        ) {
          return isLive(session);
        }

        if (
          currentFilter === "incident"
        ) {
          return (
            session.incidents?.length > 0
          );
        }

        return true;
      }
    );


  const signature =
    `${currentFilter}:${
      filtered
        .map(session =>
          JSON.stringify([
            session.id,
            session.created_at,
            session.git_commit_sha,
            session.uptime_seconds || 0,
            session.downtime_seconds || 0,
            session.incidents,
            session.telemetry?.at(-1),
            session.agentLogs?.at(-1)?.t,
            session.commands?.map(command => [command.id, command.status, command.result]),
            isLive(session)
          ])
        )
        .join("|")
    }`;


  if (
    signature ===
    lastRenderedSignature
  ) {
    return;
  }

  lastRenderedSignature =
    signature;


  if (!filtered.length) {

    els.sessionList.innerHTML = "";

    els.emptyState.hidden = false;

    els.emptyState.textContent =
      currentSessions.length
        ? "No runs match this filter."
        : "No training runs have been logged for this project yet.";

    return;
  }


  els.emptyState.hidden = true;


  const existing =
    new Map(
      [
        ...els.sessionList.children
      ].map(node => [
        node.dataset.sessionId,
        node
      ])
    );


  const fragment =
    document.createDocumentFragment();


  const reused = new Set();

  filtered.forEach(session => {

    const next =
      renderSession(session);

    const previous =
      existing.get(session.id);


    if (
      previous &&
      previous.dataset.renderSignature ===
        next.dataset.renderSignature
    ) {

      reused.add(previous);

      fragment.appendChild(previous);

      return;
    }


    if (
      previous?.classList.contains(
        "is-open"
      )
    ) {

      next.classList.add(
        "is-open"
      );

      next
        .querySelector(".session-row")
        .setAttribute(
          "aria-expanded",
          "true"
        );
    }


    if (previous) {
      destroyLossChart(previous);
    }


    fragment.appendChild(next);
  });


  // Any nodes not carried forward (session filtered out or removed)
  // still hold live Chart.js instances tied to now-detached canvases.
  existing.forEach(node => {

    if (!reused.has(node)) {
      destroyLossChart(node);
    }
  });


  els.sessionList.replaceChildren(
    fragment
  );

  els.sessionList
    .querySelectorAll(".session.is-open")
    .forEach(initLossChart);
}


/* ============================================================
   REFRESH
============================================================ */

async function refreshSessions() {

  if (!currentProject) {
    return;
  }

  // A newer refresh (another project opened, a command sent) supersedes this one.
  const generation = ++refreshGeneration;

  refreshInFlight = true;

  els.refreshBtn.classList.add(
    "is-loading"
  );

  els.refreshBtn.setAttribute(
    "aria-busy",
    "true"
  );

  try {

    const project = currentProject;

    const rows =
      await loadSessions(
        project.project_id
      );

    await loadCommands(rows);

    // Switched project, or a newer refresh started, while this was loading.
    if (project !== currentProject || generation !== refreshGeneration) {
      return;
    }

    currentSessions = rows;

    renderStats(
      currentSessions
    );

    renderSessions();

    updateNotifyBaseline(
      currentSessions
    );

    els.refreshLabel.textContent =
      "Updated just now";

  } catch (error) {

    if (generation !== refreshGeneration) {
      return;
    }

    console.error(error);

    els.refreshLabel.textContent =
      "Update failed";

  } finally {

    if (generation !== refreshGeneration) {
      return;
    }

    refreshInFlight = false;

    els.refreshBtn.classList.remove(
      "is-loading"
    );

    els.refreshBtn.removeAttribute(
      "aria-busy"
    );
  }
}


/* ============================================================
   LOGIN FLOW
============================================================ */

async function handleLogin(
  email,
  password
) {

  els.loginError.textContent = "";

  const button =
    els.loginForm.querySelector(
      "button[type=submit]"
    );

  button.disabled = true;

  button.innerHTML =
    `Signing in <span>…</span>`;


  try {

    const user =
      await loginUser(
        email,
        password
      );


    if (!user) {

      els.loginError.textContent =
        "Incorrect email or password.";

      return;
    }


    currentUser = user;

    currentAccessToken =
      user.access_token;


    sessionStorage.setItem(
      SS_SESSION_KEY,
      JSON.stringify(user)
    );


    localStorage.setItem(
      LS_EMAIL_KEY,
      email
    );


    els.loginPassword.value = "";

    await showWorkspaces();

  } catch (error) {

    console.error(error);

    els.loginError.textContent =
      "Couldn't reach Pulse. Check your connection and try again.";

  } finally {

    button.disabled = false;

    button.innerHTML =
      `Sign in <span>→</span>`;
  }
}


/* ============================================================
   EVENTS
============================================================ */

els.loginForm.addEventListener(
  "submit",
  event => {

    event.preventDefault();

    const email =
      els.loginUsername.value.trim();

    const password =
      els.loginPassword.value;

    if (!email || !password) {
      return;
    }

    handleLogin(
      email,
      password
    );
  }
);


els.homeBtn.addEventListener(
  "click",
  () => {

    if (currentUser) {
      showWorkspaces();
    } else {
      showLogin();
    }
  }
);

els.sidebarToggle.addEventListener("click", () => {
  setWorkspaceSidebarOpen(els.workspaceSidebar.hidden);
});

els.workspaceList.addEventListener("contextmenu", event => {
  const row = event.target.closest(".sidebar-workspace-item");
  if (!row) return;

  const team = workspaceRecords.get(row.dataset.teamId);
  if (!team) return;

  event.preventDefault();
  sidebarContextTeam = team;
  const canRename =
    team.owner_id === currentUser?.id || isTeamAdmin(team);
  els.sidebarContextMenu.innerHTML = `
    <button type="button" role="menuitem" data-sidebar-action="open">Open workspace</button>
    ${canRename ? `<button type="button" role="menuitem" data-sidebar-action="rename">Rename workspace</button>` : ""}
  `;
  els.sidebarContextMenu.hidden = false;
  const menu = els.sidebarContextMenu;
  const x = event.clientX || row.getBoundingClientRect().left;
  const y = event.clientY || row.getBoundingClientRect().bottom;
  menu.style.left = `${Math.min(x, window.innerWidth - menu.offsetWidth - 8)}px`;
  menu.style.top = `${Math.min(y, window.innerHeight - menu.offsetHeight - 8)}px`;
  menu.querySelector("[role=menuitem]")?.focus();
});

els.workspaceList.addEventListener("keydown", event => {
  if (!["ContextMenu", "F10"].includes(event.key) || (event.key === "F10" && !event.shiftKey)) return;
  const row = event.target.closest(".sidebar-workspace-item");
  if (!row) return;
  event.preventDefault();
  const bounds = row.getBoundingClientRect();
  row.dispatchEvent(new MouseEvent("contextmenu", {
    bubbles: true,
    clientX: bounds.left,
    clientY: bounds.bottom
  }));
});

els.sidebarContextMenu.addEventListener("click", async event => {
  const action = event.target.closest("[data-sidebar-action]")?.dataset.sidebarAction;
  if (!action || !sidebarContextTeam) return;

  const team = sidebarContextTeam;
  const row = els.workspaceList.querySelector(
    `.sidebar-workspace-item[data-team-id="${CSS.escape(String(team.team_id))}"]`
  );
  els.sidebarContextMenu.hidden = true;
  sidebarContextTeam = null;

  if (action === "open") {
    await openWorkspace(team);
  } else if (action === "rename" && row) {
    await renameWorkspace(team, row);
    row.focus();
  }
});

document.addEventListener("click", event => {
  if (!els.sidebarContextMenu.hidden && !event.target.closest("#sidebar-context-menu")) {
    els.sidebarContextMenu.hidden = true;
    sidebarContextTeam = null;
  }
});

document.addEventListener("keydown", event => {
  if (event.key === "Escape" && !els.sidebarContextMenu.hidden) {
    els.sidebarContextMenu.hidden = true;
    sidebarContextTeam = null;
  }
});


els.projectJoinForm.addEventListener(
  "submit",
  async event => {

    event.preventDefault();

    const code =
      els.projectJoinCode.value
        .trim()
        .toUpperCase();

    if (!code || !currentTeam) {
      return;
    }

    els.projectJoinMsg.textContent = "";

    try {

      const project =
        await joinSecretProject(
          currentTeam,
          code
        );

      els.projectJoinCode.value = "";

      await loadAndRenderProjects();
      await openProject(project);

    } catch (error) {

      console.warn(error);

      els.projectJoinMsg.textContent =
        "Couldn't join: that code doesn't match a secret project in this workspace.";
    }
  }
);


els.sessionList.addEventListener("click", event => {
  const button = event.target.closest(".run-picker-item");
  if (!button || button.dataset.sessionId === activeSessionId) return;
  activeSessionId = button.dataset.sessionId;
  lastRenderedSignature = "";
  renderSessions();
});


els.activeConsole.addEventListener("input", event => {
  if (event.target.matches(".run-menu-search")) {
    filterRunMenu(event.target.value);
    return;
  }
  if (!event.target.matches(".command-field")) return;
  commandDrafts.set(activeSessionId, event.target.value);
  commandSuggestionIndex = 0;
  updateCommandSuggestions();
});


els.activeConsole.addEventListener("click", event => {
  const workspaceMenuToggle = event.target.closest(".workspace-menu-toggle");
  if (workspaceMenuToggle) {
    setWorkspaceSidebarOpen(els.workspaceSidebar.hidden);
    return;
  }

  const menuToggle = event.target.closest(".run-menu-toggle");
  if (menuToggle) {
    const menu = els.activeConsole.querySelector(".run-menu");
    const isOpen = menuToggle.getAttribute("aria-expanded") === "true";
    menuToggle.setAttribute("aria-expanded", String(!isOpen));
    menu.hidden = isOpen;
    if (!isOpen) menu.querySelector(".run-menu-search")?.focus();
    return;
  }
  const pinToggle = event.target.closest(".run-pin-toggle");
  if (pinToggle) {
    togglePinnedRun(pinToggle.dataset.sessionId);
    return;
  }
  const runTab = event.target.closest(".run-tab-pill, .run-picker-item");
  if (runTab) {
    const menu = els.activeConsole.querySelector(".run-menu");
    if (menu) menu.hidden = true;
    els.activeConsole.querySelector(".run-menu-toggle")?.setAttribute("aria-expanded", "false");
    if (runTab.dataset.sessionId !== activeSessionId) {
      activeSessionId = runTab.dataset.sessionId;
      lastRenderedSignature = "";
      renderSessions();
    }
    return;
  }
  const suggestion = event.target.closest(".command-suggestion");
  if (suggestion) chooseCommandSuggestion(suggestion.dataset.command);
  const answer = event.target.closest(".question-answer");
  if (answer && !answer.disabled) sendQuestionAnswer(answer.dataset.questionId, answer.dataset.answer);
});


document.addEventListener("click", event => {
  const menu = els.activeConsole.querySelector(".run-menu");
  const toggle = els.activeConsole.querySelector(".run-menu-toggle");
  if (!menu || menu.hidden || event.target.closest(".run-tab-bar")) return;
  menu.hidden = true;
  toggle?.setAttribute("aria-expanded", "false");
});


els.activeConsole.addEventListener("keydown", event => {
  if (event.key === "Escape") {
    const menu = els.activeConsole.querySelector(".run-menu");
    if (menu && !menu.hidden) {
      menu.hidden = true;
      els.activeConsole.querySelector(".run-menu-toggle")?.setAttribute("aria-expanded", "false");
      els.activeConsole.querySelector(".run-menu-toggle")?.focus();
      event.preventDefault();
      return;
    }
  }
  const approvalDialog = event.target.closest(".run-question-backdrop .run-question");
  if (approvalDialog && !event.altKey && !event.ctrlKey && !event.metaKey) {
    const answer = event.key.toLowerCase() === "y" ? "y"
      : event.key.toLowerCase() === "n" ? "n" : null;
    const button = answer && approvalDialog.querySelector(`.question-answer[data-answer="${answer}"]`);
    if (button && !button.disabled) {
      event.preventDefault();
      button.click();
    }
    return;
  }
  if (!event.target.matches(".command-field")) return;
  const matches = matchingCommands(event.target.value.trim());
  const menu = els.activeConsole.querySelector(".command-suggestions");
  if (menu && !menu.hidden && matches.length && event.key === "ArrowDown") {
    event.preventDefault();
    commandSuggestionIndex = (commandSuggestionIndex + 1) % matches.length;
    updateCommandSuggestions();
    return;
  }
  if (menu && !menu.hidden && matches.length && event.key === "ArrowUp") {
    event.preventDefault();
    commandSuggestionIndex = (commandSuggestionIndex + matches.length - 1) % matches.length;
    updateCommandSuggestions();
    return;
  }
  if (matches.length && event.key === "Tab") {
    event.preventDefault();
    chooseCommandSuggestion(matches[commandSuggestionIndex][0]);
    return;
  }
  if (event.key === "Escape" && menu && !menu.hidden) {
    event.preventDefault();
    event.target.value = "";
    commandDrafts.set(activeSessionId, "");
    updateCommandSuggestions();
    return;
  }
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    const typed = event.target.value.trim().toLowerCase();
    if (menu && !menu.hidden && matches.length && !matches.some(([command]) => command === typed)) {
      chooseCommandSuggestion(matches[commandSuggestionIndex][0]);
      return;
    }
    event.target.form.requestSubmit();
  }
});


els.activeConsole.addEventListener("submit", async event => {

  const questionForm = event.target.closest(".question-text");
  if (questionForm) {
    event.preventDefault();
    const answer = questionForm.elements.answer.value;
    sendQuestionAnswer(questionForm.dataset.questionId, answer);
    return;
  }

  const form = event.target.closest(".command-composer");
  if (!form) return;
  event.preventDefault();

  const session = currentSessions.find(item => item.id === form.dataset.runId);
  const field = form.elements.command;
  const status = form.querySelector(".command-status");
  const button = form.querySelector("button[type=submit]");
  const command = field.value.trim();
  if (!session || !command || !currentUser) return;

  const pendingQuestion = (session.commands || []).find(item =>
    item.status === "processing" && questionOf(item)
  );
  if (pendingQuestion) {
    const question = questionOf(pendingQuestion);
    const yesNoAnswer = /^(y|yes|n|no)$/i.exec(command);
    if (!question.options && isYesNo(question.label) && yesNoAnswer) {
      sendQuestionAnswer(pendingQuestion.id, /^[yn]/i.test(yesNoAnswer[0]) ? "y" : "n");
      field.value = "";
      return;
    }
    status.textContent = "Respond to the pending question above before sending another prompt.";
    return;
  }

  button.disabled = true;
  status.textContent = "Sending to runner…";
  try {
    await pgPost("Commands", {
      run_id: session.id,
      project_id: session.project_id,
      user_id: currentUser.id,
      command
    });
    field.value = "";
    session.commands = session.commands || [];
    session.commands.push({
      id: `local-${Date.now()}`,
      run_id: session.id,
      user_id: currentUser.id,
      command,
      status: "pending",
      created_at: new Date().toISOString()
    });
    commandDrafts.set(session.id, "");
    lastRenderedSignature = "";
    renderSessions();
    refreshSessions();
  } catch (error) {
    console.error("Could not queue command:", error);
    status.textContent = "Could not send. Check the Commands table policies and try again.";
    button.disabled = false;
  }
});


els.refreshBtn.addEventListener(
  "click",
  refreshSessions
);


els.notificationsBtn.addEventListener(
  "click",
  openNotificationsModal
);

els.notificationsCloseBtn.addEventListener(
  "click",
  closeNotificationsModal
);

els.notificationsModal.addEventListener(
  "click",
  event => {

    if (event.target === els.notificationsModal) {
      closeNotificationsModal();
    }
  }
);

els.notificationsSaveBtn.addEventListener(
  "click",
  handleSaveNotifications
);

els.notificationsTestBtn.addEventListener(
  "click",
  handleTestNotifications
);


/* ============================================================
   RESTORE LOGIN
============================================================ */

// Read (and immediately strip) any ?integration=&status= params Discord's/
// Slack's OAuth callback redirected back with, before anything else runs.
const oauthReturnParams = (() => {

  const params =
    new URLSearchParams(window.location.search);

  const integration = params.get("integration");
  const status = params.get("status");

  if (integration) {

    window.history.replaceState(
      null,
      "",
      window.location.pathname
    );

    return { integration, status };
  }

  return null;
})();


// After an OAuth redirect brings the browser back, the page has fully
// reloaded -- currentTeam is gone even though currentUser was restored
// from sessionStorage below. This reopens whichever workspace the user
// was connecting a provider for, then surfaces the result in the
// notifications modal.
async function resumeAfterOAuthRedirect() {

  const pendingTeamId =
    sessionStorage.getItem(SS_PENDING_OAUTH_TEAM_KEY);

  sessionStorage.removeItem(
    SS_PENDING_OAUTH_TEAM_KEY
  );

  if (
    !oauthReturnParams ||
    !currentUser ||
    !pendingTeamId
  ) {
    return;
  }

  try {

    const teams =
      await loadWorkspacesForUser(currentUser.id);

    const team =
      teams.find(
        candidate =>
          String(candidate.team_id) === String(pendingTeamId)
      );

    if (!team) return;

    await openWorkspace(team);

    pendingIntegrationNotice = oauthReturnParams;

    await openNotificationsModal();

  } catch (error) {

    console.warn(
      "Could not resume workspace after OAuth redirect:",
      error
    );
  }
}


loadNotificationSettings();


const lastEmail =
  localStorage.getItem(
    LS_EMAIL_KEY
  );

if (lastEmail) {
  els.loginUsername.value =
    lastEmail;
}


try {

  const saved =
    JSON.parse(
      sessionStorage.getItem(
        SS_SESSION_KEY
      ) || "null"
    );


  if (
    saved &&
    saved.id &&
    saved.email &&
    saved.access_token
  ) {

    currentUser = saved;

    currentAccessToken =
      saved.access_token;

    currentRefreshToken =
      saved.refresh_token || null;

    showWorkspaces().then(
      resumeAfterOAuthRedirect
    );
  }

} catch {
  // Invalid saved session.
}


/* ============================================================
   AUTO REFRESH
============================================================ */

setInterval(
  () => {

    if (
      !els.viewDashboard.hidden
    ) {

      if (
        els.refreshLabel.textContent ===
        "Updated just now"
      ) {
        els.refreshLabel.textContent =
          "Updating...";
      }

      refreshSessions();
    }

  },
  30000
);


// A new question from Pulse: a browser notification (if they are on), once per question.
const announcedQuestions = new Set();

function announceQuestions(sessions) {
  sessions.forEach(session => (session.commands || []).forEach(command => {
    const asked = questionOf(command);
    if (!asked || command.status !== "processing" || announcedQuestions.has(command.id)) return;
    announcedQuestions.add(command.id);
    if (notificationSettings.browserPush) {
      sendBrowserNotification(`Pulse is asking (${runLabel(session)})`, asked.label.slice(0, 200), `pulse-ask-${command.id}`);
    }
  }));
}

// The live view: what Pulse is doing right now, drawn in place (the rest of the console is not
// redrawn, so nothing typed or scrolled is lost) every time the live row changes.
let lastLiveHtml = "";

function renderLive() {
  const box = els.activeConsole.querySelector(".run-live");
  const session = currentSessions.find(item => item.id === activeSessionId);
  if (!box || !session) return;
  const state = liveStateOf(session);

  const run = state?.run;
  if (run) {
    const stepEl = els.activeConsole.querySelector("[data-live-step]");
    if (stepEl && run.step !== undefined) stepEl.textContent = metricValue(run.step);
    const picker = [...els.sessionList.querySelectorAll("[data-live-picker]")].find(node => node.dataset.livePicker === session.id);
    const loss = ["loss", "train_loss", "loss_value", "current_loss"].map(name => run.values?.[name]).find(value => value !== undefined);
    if (picker && run.step !== undefined) {
      picker.innerHTML = `step ${escapeHtml(metricValue(run.step))} <i>·</i> loss ${escapeHtml(metricValue(loss ?? findMetric(session, ["loss", "train_loss", "loss_value", "current_loss"])))}`;
    }
    Object.entries(run.values || {}).forEach(([name, value]) => {
      const el = [...els.activeConsole.querySelectorAll("[data-live-metric]")].find(node => node.dataset.liveMetric === name);
      if (el) el.textContent = metricValue(value);
    });
  }

  let html = "";
  if (state?.busy) {
    const lines = (state.activity || []).map(item => {
      const text = renderCliText(item.ansi || item.text || "");
      const streaming = item.live ? `<span class="live-caret" aria-hidden="true"></span>` : "";
      switch (item.kind) {
        case "thinking": return `<div class="live-line is-thinking">${text}${streaming}</div>`;
        case "tool":
        case "edit": return `<div class="live-line is-tool">${text}</div>`;
        case "user": return `<div class="live-line is-user">${text}</div>`;
        case "note": return `<div class="live-line is-note">${text}</div>`;
        case "error": return `<div class="live-line is-error">${text}</div>`;
        default: return `<div class="live-line">${text}${streaming}</div>`;
      }
    }).join("");
    const doing = state.asking ? "waiting for an answer" : (state.doing || "working");
    html = `
      <article class="console-exchange is-live">
        <div class="console-speaker">PULSE <span class="live-dot" aria-hidden="true"></span> <span class="live-doing">${escapeHtml(doing)}</span></div>
        ${lines || `<div class="live-line is-note">starting…</div>`}
      </article>
    `;
  }
  if (html === lastLiveHtml && box.innerHTML.trim() === html.trim()) return;
  const transcript = box.closest(".run-transcript");
  const atBottom = !transcript || transcript.scrollHeight - transcript.clientHeight - transcript.scrollTop < 48;
  box.innerHTML = html;
  lastLiveHtml = html;
  if (transcript && atBottom) transcript.scrollTop = transcript.scrollHeight;
}


// The run on screen is polled every second while its Pulse is there (the live view, a
// question coming up, a prompt sent from here); other runs with a prompt on its way every
// 3 s. The full refresh is every 30 s.
let livePollInFlight = false;
let livePollTick = 0;
const liveWasBusy = new Map();

async function pollOpenRows(session) {
  const rows = await pgGet("Commands", {
    run_id: `eq.${session.id}`,
    status: "in.(pending,processing)",
    select: "id,run_id,user_id,command,status,result,created_at,completed_at",
    order: "created_at.asc",
    limit: "50"
  });
  const before = session.commands || [];
  const isOpen = command => ["pending", "processing"].includes(command.status) && !String(command.id).startsWith("local-");
  const vanished = before.filter(isOpen).some(command => !rows.some(row => row.id === command.id));
  if (vanished) {
    await loadCommands([session]);                 // something finished: its final state
  } else {
    session.commands = before.filter(command => !isOpen(command))
      .filter(command => !String(command.id).startsWith("local-") || !rows.some(row => row.command === command.command))
      .concat(rows)
      .sort((a, b) => Date.parse(a.created_at || 0) - Date.parse(b.created_at || 0));
  }
}

setInterval(
  async () => {

    if (els.viewDashboard.hidden || livePollInFlight || !currentProject) return;
    livePollTick++;

    const hasOpen = session => (session.commands || []).some(command =>
      ["pending", "processing"].includes(command.status));
    const active = currentSessions.find(session => session.id === activeSessionId);
    const others = livePollTick % 3 === 0
      ? currentSessions.filter(session => session !== active && (session.commands || []).some(command =>
          ["pending", "processing"].includes(command.status) && !isLiveRow(command)))
      : [];
    const pollActive = active && (isLive(active) || hasOpen(active));
    if (!pollActive && !others.length) return;

    livePollInFlight = true;
    const project = currentProject;
    try {
      const before = lastRenderedSignature;
      if (pollActive) await pollOpenRows(active);
      if (others.length) await loadCommands(others);
      if (project !== currentProject) return;
      announceQuestions([active, ...others].filter(Boolean));
      renderSessions();                              // redraws only when something besides the live view changed
      if (lastRenderedSignature === before) renderLive();
      if (active) {
        // a turn just ended: its answer is in the run's agent log now
        const busy = Boolean(liveStateOf(active)?.busy);
        if (liveWasBusy.get(active.id) && !busy) refreshSessions();
        liveWasBusy.set(active.id, busy);
      }
    } catch (error) {
      console.warn("Live update failed:", error);
    } finally {
      livePollInFlight = false;
    }
  },
  1000
);


setInterval(
  () => {

    if (
      currentUser &&
      currentTeam
    ) {
      loadAndRenderProjects();
    }

  },
  15000
);


/* ============================================================
   INITIAL VIEW
============================================================ */

if (!currentUser) {
  showLogin();
}


function runLabel(session) {
  const script = session.env?.script || session.env?.pulse_session || session.script_name;
  const name = script ? String(script).split(/[\\/]/).pop() : `run ${String(session.id || "").slice(0, 8)}`;
  const duplicates = currentSessions.filter(item => {
    const itemScript = item.env?.script || item.env?.pulse_session || item.script_name;
    const itemName = itemScript
      ? String(itemScript).split(/[\\/]/).pop()
      : `run ${String(item.id || "").slice(0, 8)}`;
    return itemName === name;
  });
  return duplicates.length > 1 ? `${name} · ${String(session.id || "").slice(0, 8)}` : name;
}


function ensurePinnedRunsLoaded() {
  const projectId = String(currentProject?.project_id || "");
  if (pinnedRunsProjectId === projectId) return;
  pinnedRunsProjectId = projectId;
  pinnedRunIds = new Set();
  unpinnedLiveRunIds = new Set();
  observedLiveRunIds = new Set();
  if (!projectId) return;
  const saved = localStorage.getItem(`${LS_PINNED_RUNS_PREFIX}${projectId}`);
  if (saved) {
    try {
      const ids = JSON.parse(saved);
      if (!Array.isArray(ids) || ids.some(id => typeof id !== "string")) {
        throw new TypeError("Pinned run data must be an array of run IDs.");
      }
      pinnedRunIds = new Set(ids);
    } catch (error) {
      console.warn("Could not load pinned runs:", error);
    }
  }
  const unpinnedLive = localStorage.getItem(`${LS_UNPINNED_LIVE_RUNS_PREFIX}${projectId}`);
  if (unpinnedLive) {
    try {
      const ids = JSON.parse(unpinnedLive);
      if (!Array.isArray(ids) || ids.some(id => typeof id !== "string")) {
        throw new TypeError("Unpinned live run data must be an array of run IDs.");
      }
      unpinnedLiveRunIds = new Set(ids);
    } catch (error) {
      console.warn("Could not load unpinned live runs:", error);
    }
  }
}


function togglePinnedRun(sessionId) {
  if (!sessionId || !currentProject?.project_id) return;
  ensurePinnedRunsLoaded();
  const existingMenu = els.activeConsole.querySelector(".run-menu");
  const wasMenuOpen = Boolean(existingMenu && !existingMenu.hidden);
  const searchQuery = els.activeConsole.querySelector(".run-menu-search")?.value || "";
  const session = currentSessions.find(item => item.id === sessionId);
  if (pinnedRunIds.has(sessionId)) {
    pinnedRunIds.delete(sessionId);
    if (session && isLive(session)) unpinnedLiveRunIds.add(sessionId);
  } else {
    pinnedRunIds.add(sessionId);
    unpinnedLiveRunIds.delete(sessionId);
  }
  localStorage.setItem(
    `${LS_PINNED_RUNS_PREFIX}${currentProject.project_id}`,
    JSON.stringify([...pinnedRunIds])
  );
  localStorage.setItem(
    `${LS_UNPINNED_LIVE_RUNS_PREFIX}${currentProject.project_id}`,
    JSON.stringify([...unpinnedLiveRunIds])
  );
  lastRenderedSignature = "";
  renderSessions();
  if (wasMenuOpen) {
    const menu = els.activeConsole.querySelector(".run-menu");
    const toggle = els.activeConsole.querySelector(".run-menu-toggle");
    const search = els.activeConsole.querySelector(".run-menu-search");
    if (menu && toggle && search) {
      menu.hidden = false;
      toggle.setAttribute("aria-expanded", "true");
      search.value = searchQuery;
      search.dispatchEvent(new Event("input", { bubbles: true }));
      search.focus();
    }
  }
}


function filterRunMenu(value) {
  const query = value.trim().toLowerCase();
  els.activeConsole.querySelectorAll(".run-menu-item").forEach(item => {
    item.hidden = !item.textContent.toLowerCase().includes(query);
  });
}


function renderRunPickerItem(session) {
  const live = isLive(session);
  const loss = findMetric(session, ["loss", "train_loss", "loss_value", "current_loss"]);
  const step = findMetric(session, ["step", "global_step", "epoch"]);
  const selected = session.id === activeSessionId;
  const pinned = pinnedRunIds.has(session.id);
  const label = runLabel(session);
  return `
    <div class="run-menu-item ${selected ? "is-selected" : ""}" role="listitem">
      <button class="run-picker-item ${live ? "is-live-run" : "is-stale-run"}" type="button"
        data-session-id="${escapeHtml(session.id)}" aria-pressed="${selected}" aria-label="Open ${escapeHtml(label)}" title="${escapeHtml(label)}">
        <span class="run-picker-dot ${live ? "is-live" : ""}" aria-hidden="true"></span>
        <span class="run-picker-copy">
          <strong>${escapeHtml(label)}</strong>
          <span data-live-picker="${escapeHtml(session.id)}">${live ? "Live" : "Past run"} <i>·</i> step ${escapeHtml(metricValue(step))} <i>·</i> loss ${escapeHtml(metricValue(loss))}</span>
        </span>
      </button>
      <button class="run-pin-toggle ${pinned ? "is-pinned" : ""}" type="button" data-session-id="${escapeHtml(session.id)}"
        aria-pressed="${pinned}" aria-label="${pinned ? "Unpin" : "Pin"} ${escapeHtml(label)}" title="${pinned ? "Unpin run" : "Pin run"}">
        <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M10.8 1.8 14 5l-2.2.5-2.1 2.1.4 2.4-.8.8-2.5-2.5-3.5 3.5-.8-.8L6 7.5 3.5 5l.8-.8 2.4.4 2.1-2.1.5-2.2Z" /></svg>
      </button>
    </div>
  `;
}


function renderRunTab(session) {
  const live = isLive(session);
  const selected = session.id === activeSessionId;
  const label = runLabel(session);
  return `
    <button class="run-tab-pill ${selected ? "is-selected" : ""} ${live ? "is-live" : ""}" type="button"
      data-session-id="${escapeHtml(session.id)}" aria-pressed="${selected}" aria-label="Open ${escapeHtml(label)}" title="${escapeHtml(label)}">
      <span class="run-picker-dot ${live ? "is-live" : ""}" aria-hidden="true"></span>
      <span>${escapeHtml(label)}</span>
    </button>
  `;
}


function renderSessions() {

  ensurePinnedRunsLoaded();
  const liveRuns = currentSessions.filter(isLive);
  let defaultPinsChanged = false;
  for (const run of liveRuns) {
    if (observedLiveRunIds.has(run.id)) continue;
    observedLiveRunIds.add(run.id);
    if (!unpinnedLiveRunIds.has(run.id) && !pinnedRunIds.has(run.id)) {
      pinnedRunIds.add(run.id);
      defaultPinsChanged = true;
    }
  }
  if (defaultPinsChanged && currentProject?.project_id) {
    localStorage.setItem(
      `${LS_PINNED_RUNS_PREFIX}${currentProject.project_id}`,
      JSON.stringify([...pinnedRunIds])
    );
  }
  if (autoSelectLiveRun) {
    const initialRun = liveRuns[0] || currentSessions[0];
    activeSessionId = initialRun?.id || null;
    autoSelectLiveRun = false;
  } else if (!currentSessions.some(session => session.id === activeSessionId)) {
    activeSessionId = (liveRuns[0] || currentSessions[0])?.id || null;
  }

  const signature = JSON.stringify([
    activeSessionId,
    currentSessions.map(session => [
      session.id,
      session.created_at,
      session.git_commit_sha,
      session.uptime_seconds,
      session.downtime_seconds,
      session.incidents?.length || 0,
      session.telemetry?.at(-1),
      session.agentLogs?.at(-1)?.t,
      session.commands?.map(command => isLiveRow(command) ? [command.id, command.status] : [command.id, command.status, command.result]),
      isLive(session)
    ])
  ]);
  if (signature === lastRenderedSignature) return;
  lastRenderedSignature = signature;

  renderStats(currentSessions);
  els.emptyState.hidden = currentSessions.length > 0;
  els.emptyState.textContent = "No runs have been logged for this project yet.";
  els.sessionList.replaceChildren();

  const session = currentSessions.find(item => item.id === activeSessionId);
  const oldField = els.activeConsole.querySelector(".command-field");
  const wasFocused = oldField && document.activeElement === oldField;
  const oldTranscript = els.activeConsole.querySelector(".run-transcript");
  const oldTranscriptScroll = oldTranscript?.scrollTop || 0;
  const transcriptAtBottom = !oldTranscript ||
    oldTranscript.scrollHeight - oldTranscript.clientHeight - oldTranscript.scrollTop < 32;
  const draft = commandDrafts.get(activeSessionId) || "";
  const oldRunMenu = els.activeConsole.querySelector(".run-menu");
  const runMenuWasOpen = oldRunMenu && !oldRunMenu.hidden;
  const runMenuSearchValue = els.activeConsole.querySelector(".run-menu-search")?.value || "";
  const visibleRuns = currentSessions.filter(item =>
    item.id === activeSessionId || pinnedRunIds.has(item.id)
  );
  const hiddenRunCount = currentSessions.length - visibleRuns.length;
  const runContent = session
    ? renderRunWorkspace(session)
    : `<div class="console-empty-state"><span>PULSE / DEBUG</span><p>${currentProject ? "Select a run to open its console." : "Select a workspace and project to open its training runs."}</p></div>`;

  els.activeConsole.innerHTML = `
      <div class="run-window-shell">
        <nav class="run-tab-bar" aria-label="Training runs">
          ${renderWorkspaceMenuToggle()}
          <div class="run-tab-pills">${visibleRuns.map(renderRunTab).join("")}</div>
          ${currentSessions.length ? `
            <button class="run-menu-toggle" type="button" aria-haspopup="dialog" aria-expanded="false"
              aria-label="Browse all ${currentSessions.length} runs, ${hiddenRunCount} hidden">+${hiddenRunCount}
            </button>
            <section class="run-menu" role="dialog" aria-label="All runs" hidden>
              <header class="run-menu-header">
                <div><strong>All runs</strong><span>${currentSessions.length} runs</span></div>
                <input class="run-menu-search" type="search" placeholder="Find a run..." aria-label="Find a run">
              </header>
              <div class="run-menu-list" role="list">${currentSessions.map(item => renderRunPickerItem(item)).join("")}</div>
            </section>
          ` : ""}
        </nav>
        <div class="run-window-layout">
          <section class="run-window-content">${runContent}</section>
        </div>
        </div>`;

  mountWorkspaceNavigation(els.activeConsole.querySelector(".run-window-shell"));

  const runMenu = els.activeConsole.querySelector(".run-menu");
  const runMenuToggle = els.activeConsole.querySelector(".run-menu-toggle");
  const runMenuSearch = els.activeConsole.querySelector(".run-menu-search");
  if (runMenuWasOpen && runMenu && runMenuToggle && runMenuSearch) {
    runMenu.hidden = false;
    runMenuToggle.setAttribute("aria-expanded", "true");
    runMenuSearch.value = runMenuSearchValue;
    filterRunMenu(runMenuSearchValue);
  }

  const approvalDialog = els.activeConsole.querySelector(".run-question-backdrop .run-question");
  if (approvalDialog) {
    const defaultAnswer = approvalDialog.querySelector('.question-answer[data-answer="n"]:not(:disabled)');
    (defaultAnswer || approvalDialog.querySelector(".question-answer:not(:disabled)") || approvalDialog).focus();
  }

  const transcript = els.activeConsole.querySelector(".run-transcript");
  if (transcript) {
    transcript.scrollTop = transcriptAtBottom ? transcript.scrollHeight : oldTranscriptScroll;
  }

  renderLive();

  const field = els.activeConsole.querySelector(".command-field");
  if (field) {
    field.value = draft;
    if (wasFocused && !approvalDialog) field.focus();
    updateCommandSuggestions();
  }
}


function matchingCommands(text) {
  if (!text.startsWith("/") || text.includes(" ")) return [];
  const source = [...DEBUG_COMMANDS, ...HOME_COMMANDS];
  const unique = new Map();
  source.forEach(command => {
    if (!unique.has(command[0])) unique.set(command[0], command);
  });
  return [...unique.values()].filter(command => command[0].startsWith(text.toLowerCase())).slice(0, 7);
}


function updateCommandSuggestions() {
  const field = els.activeConsole.querySelector(".command-field");
  const menu = els.activeConsole.querySelector(".command-suggestions");
  if (!field || !menu) return;
  const matches = matchingCommands(field.value.trim());
  commandSuggestionIndex = Math.min(commandSuggestionIndex, Math.max(0, matches.length - 1));
  menu.innerHTML = matches.map(([command, description], index) => `
    <button class="command-suggestion ${index === commandSuggestionIndex ? "is-active" : ""}"
      type="button" role="option" aria-selected="${index === commandSuggestionIndex}"
      data-command="${escapeHtml(command)}">
      <code>${escapeHtml(command)}</code><span>${escapeHtml(description)}</span>
    </button>
  `).join("");
  menu.hidden = matches.length === 0;
}


function chooseCommandSuggestion(command) {
  const field = els.activeConsole.querySelector(".command-field");
  if (!field) return;
  field.value = `${command} `;
  commandDrafts.set(activeSessionId, field.value);
  commandSuggestionIndex = 0;
  field.focus();
  updateCommandSuggestions();
}