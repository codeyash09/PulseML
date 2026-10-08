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
let currentUser = null;
let currentTeam = null;
let currentProject = null;
let currentSessions = [];

let activeSessionId = null;
let commandDrafts = new Map();
let commandSuggestionIndex = 0;

let emailCache = new Map();

let refreshInFlight = false;
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

const HOME_COMMANDS = [
  ["/monitor", "pick a run on this machine and open it beside the agent"],
  ["/change", "look at another run; the one you leave stays watched"],
  ["/runs", "the same list: every run on this machine"],
  ["/run", "start a script under Pulse and watch it: /run train.py --epochs 3"],
  ["/agent", "switch AI provider/model (or sign in with OpenRouter)"],
  ["/files", "what the agent sees in full, and how much it can search"],
  ["/add", "put files or folders in focus"],
  ["/drop", "take a file out of focus"],
  ["/review", "show a diff and ask before applying: /review on|off"],
  ["/undo", "undo the latest change"],
  ["/log", "the change history"],
  ["/cloud", "sign-in, workspace and sync status"],
  ["/config", "your settings, remembered between starts: /config mouse off, /config agent"],
  ["/copy", "copy the agent's last answer to the clipboard (/copy 2: the one before)"],
  ["/mouse", "clicks open folded lines; /mouse off gives the mouse back to select text"],
  ["/help", "every command"],
  ["/exit", "leave Pulse"]
];

const DEBUG_COMMANDS = [
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
  ["/change", "look at another run; the one you leave stays watched"],
  ["/close", "stop watching the open run"]
];


/* ============================================================
   ELEMENTS
============================================================ */

const els = {
  viewLogin: document.getElementById("view-login"),
  viewWorkspaces: document.getElementById("view-workspaces"),
  viewProjects: document.getElementById("view-projects"),
  viewDashboard: document.getElementById("view-dashboard"),

  loginForm: document.getElementById("login-form"),
  loginUsername: document.getElementById("login-username"),
  loginPassword: document.getElementById("login-password"),
  loginError: document.getElementById("login-error"),

  homeBtn: document.getElementById("home-btn"),
  topbarRight: document.getElementById("topbar-right"),
  topbarWorkspace: document.getElementById("topbar-workspace"),

  workspaceList: document.getElementById("workspace-list"),
  workspacesEmpty: document.getElementById("workspaces-empty"),

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
  projectsNotificationsBtn: document.getElementById("projects-notifications-btn"),
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

async function pgGet(table, params = {}) {

  const url =
    new URL(`${SUPABASE_URL}/rest/v1/${table}`);

  Object.entries(params).forEach(([key, value]) => {
    url.searchParams.set(key, value);
  });

  const response = await fetch(url.toString(), {
    headers: {
      apikey: SUPABASE_KEY,
      Authorization:
        `Bearer ${currentAccessToken || SUPABASE_KEY}`
    }
  });

  if (!response.ok) {

    const text =
      await response.text().catch(() => "");

    throw new Error(
      `GET ${table} -> HTTP ${response.status}: ${text}`
    );
  }

  return response.json();
}


async function pgPatch(table, match, body) {

  const url =
    new URL(`${SUPABASE_URL}/rest/v1/${table}`);

  Object.entries(match).forEach(([key, value]) => {
    url.searchParams.set(key, value);
  });

  const response = await fetch(url.toString(), {
    method: "PATCH",

    headers: {
      apikey: SUPABASE_KEY,
      Authorization:
        `Bearer ${currentAccessToken || SUPABASE_KEY}`,
      "Content-Type": "application/json",
      Prefer: "return=minimal"
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
}


async function pgPost(table, body) {

  const response = await fetch(
    `${SUPABASE_URL}/rest/v1/${table}`,
    {
      method: "POST",
      headers: {
        apikey: SUPABASE_KEY,
        Authorization: `Bearer ${currentAccessToken || SUPABASE_KEY}`,
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

  const response = await fetch(url.toString(), {
    method,

    headers: {
      apikey: SUPABASE_KEY,
      Authorization:
        `Bearer ${currentAccessToken || SUPABASE_KEY}`,
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
        authData.access_token
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
      select: "id,run_id,user_id,command,status,result,created_at",
      order: "created_at.asc",
      limit: "200"
    });
    const byRun = new Map();
    rows.forEach(row => {
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


/* ============================================================
   FORMATTING
============================================================ */

function escapeHtml(value) {

  const element =
    document.createElement("div");

  element.textContent =
    value ?? "";

  return element.innerHTML;
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

    return;
  }

  const dashboardOpen =
    !els.viewDashboard.hidden;

  const pieces = [];

  pieces.push(
    `<span>${escapeHtml(currentUser.email)}</span>`
  );

  const projectsOpen =
    !els.viewProjects.hidden;

  if (dashboardOpen) {

    pieces.push(
      `<button
        type="button"
        id="switch-project-btn"
      >
        Switch project
      </button>`
    );
  }

  if (dashboardOpen || projectsOpen) {

    pieces.push(
      `<button
        type="button"
        id="switch-workspace-btn"
      >
        Switch workspace
      </button>`
    );
  }

  pieces.push(
    `<button
      type="button"
      id="sign-out-btn"
    >
      Sign out
    </button>`
  );

  els.topbarRight.innerHTML =
    pieces.join("");

  const switchButton =
    document.getElementById(
      "switch-workspace-btn"
    );

  if (switchButton) {

    switchButton.addEventListener(
      "click",
      showWorkspaces
    );
  }

  const switchProjectButton =
    document.getElementById(
      "switch-project-btn"
    );

  if (switchProjectButton) {

    switchProjectButton.addEventListener(
      "click",
      () => showProjects(currentTeam)
    );
  }

  document
    .getElementById("sign-out-btn")
    .addEventListener(
      "click",
      signOut
    );

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


/* ============================================================
   VIEWS
============================================================ */

function showLogin() {

  els.viewLogin.hidden = false;
  els.viewWorkspaces.hidden = true;
  els.viewProjects.hidden = true;
  els.viewDashboard.hidden = true;

  els.topbarRight.innerHTML = "";
  els.topbarWorkspace.textContent = "";
}


async function showWorkspaces() {

  els.viewLogin.hidden = true;
  els.viewWorkspaces.hidden = false;
  els.viewProjects.hidden = true;
  els.viewDashboard.hidden = true;

  currentTeam = null;
  currentProject = null;

  renderTopbarRight();

  await loadAndRenderWorkspaces();
}


function showDashboard() {

  els.viewLogin.hidden = true;
  els.viewWorkspaces.hidden = true;
  els.viewProjects.hidden = true;
  els.viewDashboard.hidden = false;

  renderTopbarRight();
}


function signOut() {

  currentUser = null;
  currentTeam = null;
  currentProject = null;
  currentSessions = [];
  currentAccessToken = null;

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

  const isAdmin =
    (team.admin_ids || [])
      .includes(currentUser.id);

  const isOwner =
    team.owner_id === currentUser.id;

  const role =
    isOwner
      ? "owner"
      : isAdmin
        ? "admin"
        : "member";

  // A plain div, not a button -- admins/owners get a real nested
  // <button> for renaming (see below), and a <button> can't contain
  // another interactive button per the HTML spec.
  const row =
    document.createElement("div");

  row.className = "workspace-row";
  row.setAttribute("role", "button");
  row.setAttribute("tabindex", "0");

  row.innerHTML = `

    <span class="workspace-main">

      <span class="workspace-name">
        ${escapeHtml(
          getWorkspaceName(team)
        )}
      </span>

      <span class="workspace-meta">

        <span>
          ${(team.members || []).length}
          member${
            (team.members || []).length === 1
              ? ""
              : "s"
          }
        </span>

        <span>
          ${escapeHtml(
            team.plan || "free"
          )} plan
        </span>

        ${
          liveCount > 0
            ? `
              <span class="workspace-live">
                <span class="workspace-live-dot"></span>
                ${liveCount}
                LIVE
                ${liveCount === 1 ? "RUN" : "RUNS"}
              </span>
            `
            : `
              <span class="workspace-no-live">
                No live runs
              </span>
            `
        }

      </span>

    </span>

    <span
      class="workspace-role ${
        isAdmin ? "is-admin" : ""
      }"
    >
      ${role}
    </span>

  `;

  row.addEventListener(
    "click",
    () => openWorkspace(team)
  );

  // Keyboard equivalent of the <button> activation this div gave up.
  row.addEventListener(
    "keydown",
    event => {

      if (event.key === "Enter" || event.key === " ") {

        event.preventDefault();

        openWorkspace(team);
      }
    }
  );

  if (isAdmin || isOwner) {

    const renameBtn =
      document.createElement("button");

    renameBtn.type = "button";
    renameBtn.className = "icon-btn workspace-rename-btn";
    renameBtn.title = "Rename workspace";

    renameBtn.innerHTML =
      `<span aria-hidden="true">✏️</span> Rename`;

    renameBtn.addEventListener(
      "click",
      event => {

        // Don't also trigger the row's own click (which would open
        // the workspace).
        event.stopPropagation();

        renameWorkspace(team, row);
      }
    );

    row.appendChild(renameBtn);
  }

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
      { name: trimmed }
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
    row.querySelector(".workspace-name");

  if (nameEl) {

    nameEl.textContent =
      getWorkspaceName(team);
  }

  // Keep the topbar/crumbs in sync if this is the open workspace.
  if (currentTeam && currentTeam.team_id === team.team_id) {

    currentTeam.name = trimmed;

    renderTopbarRight();
  }
}

async function loadAndRenderWorkspaces() {

  els.workspaceList.innerHTML = "";

  els.workspacesEmpty.hidden = true;

  try {

    const teams =
      await loadWorkspacesForUser(
        currentUser.id
      );

    if (!teams.length) {

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


    workspaceStates.forEach(
      ({ team, liveCount }) => {

        els.workspaceList.appendChild(
          renderWorkspaceRow(
            team,
            liveCount
          )
        );
      }
    );

  } catch (error) {

    console.error(error);

    els.workspacesEmpty.hidden = false;

    els.workspacesEmpty.textContent =
      "Couldn't load your workspaces. Try refreshing.";
  }
}

async function openWorkspace(team) {

  currentTeam = team;
  currentProject = null;

  // Fresh workspace -- don't show a stale integration status from
  // whatever workspace was open before. (Notifications are configured
  // per workspace, so they load here, not per project.)
  notifyBaseline = null;
  teamIntegration = null;

  await showProjects(team);

  try {
    await loadTeamIntegration();
  } catch (error) {
    console.warn("Could not load notification integrations:", error);
  }
}


function renderProjectRow(
  project,
  liveCount = 0
) {

  const repo =
    getRepoName(project.repo);

  const row =
    document.createElement("button");

  row.type = "button";
  row.className = "workspace-row";

  row.innerHTML = `

    <span class="workspace-main">

      <span class="workspace-name">
        ${escapeHtml(getProjectName(project))}
      </span>

      <span class="workspace-meta">

        ${
          repo
            ? `<span>${escapeHtml(repo)}</span>`
            : ""
        }

        ${
          liveCount > 0
            ? `
              <span class="workspace-live">
                <span class="workspace-live-dot"></span>
                ${liveCount}
                LIVE
                ${liveCount === 1 ? "RUN" : "RUNS"}
              </span>
            `
            : `
              <span class="workspace-no-live">
                No live runs
              </span>
            `
        }

      </span>

    </span>

    <span
      class="workspace-role ${
        project.is_secret ? "is-admin" : ""
      }"
    >
      ${project.is_secret ? "secret" : "project"}
    </span>

  `;

  row.addEventListener(
    "click",
    () => openProject(project)
  );

  return row;
}


async function loadAndRenderProjects() {

  const team = currentTeam;

  els.projectList.innerHTML = "";

  els.projectsEmpty.hidden = true;

  try {

    const projects =
      await loadProjectsForTeam(team);

    // Switched workspace while this was loading.
    if (team !== currentTeam) {
      return;
    }

    if (!projects.length) {

      els.projectsEmpty.hidden = false;

      return;
    }

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

    projects.forEach(project => {

      els.projectList.appendChild(
        renderProjectRow(
          project,
          liveCounts.get(project.project_id) || 0
        )
      );
    });

  } catch (error) {

    console.error(error);

    els.projectsEmpty.hidden = false;

    els.projectsEmpty.textContent =
      "Couldn't load this workspace's projects. Try refreshing.";
  }
}


async function showProjects(team) {

  if (!team) {

    await showWorkspaces();

    return;
  }

  currentTeam = team;
  currentProject = null;

  els.viewLogin.hidden = true;
  els.viewWorkspaces.hidden = true;
  els.viewDashboard.hidden = true;
  els.viewProjects.hidden = false;

  els.projectsWorkspace.textContent =
    getWorkspaceName(team).toUpperCase();

  els.projectJoinMsg.textContent = "";

  renderTopbarRight();

  await loadAndRenderProjects();
}


async function openProject(project) {

  currentProject = project;

  currentSessions = [];

  lastRenderedSignature = "";

  // Fresh project -- don't fire notifications for runs that already
  // existed before this dashboard session opened it.
  notifyBaseline = null;

  els.workspaceTitle.textContent =
    getProjectName(project);

  els.projectCrumb.textContent =
    `${getWorkspaceName(currentTeam)} / ${project.is_secret ? "SECRET PROJECT" : "PROJECT"}`
      .toUpperCase();

  showDashboard();

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

  const rows =
    await pgGet("team_integrations", {
      team_id: `eq.${currentTeam.team_id}`,

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

  if (els.projectsNotificationsBtn) {

    els.projectsNotificationsBtn.classList.toggle(
      "is-configured",
      configured
    );
  }
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
  body
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
      tag: "pulse-notifications"
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
            .replace(/^#\s*/, "")
            .trim()
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
    }
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
      }
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

        ok:
          true
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
        borderColor: "#0071e3",
        backgroundColor: "rgba(0, 113, 227, 0.05)",
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
  const commands = session.commands || [];
  const step = findMetric(session, ["step", "global_step", "epoch"]);
  const live = isLive(session);
  const telemetry = session.telemetry || [];
  const latest = latestTelemetry(session);
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
        <strong>${escapeHtml(metricValue(values.at(-1)))}</strong>
        <span class="run-metric-spark" aria-label="Recent ${escapeHtml(name)} values">${escapeHtml(metricSparkline(values))}</span>
      </div>
    `;
  }).join("");
  const latestRunStatus = [...telemetry]
    .reverse()
    .find(entry => entry.status)?.status || [...(session.incidents || [])]
      .reverse()
      .find(incident => String(incident.kind || "").startsWith("run_"))?.status;
  const runStatus = latestRunStatus || (live ? "live" : "stopped");
  const stepRate = telemetryStepRate(telemetry);
  const startedAt = session.created_at
    ? Date.parse(session.created_at) / 1000
    : Number(telemetry[0]?.t || 0);
  const elapsed = live && startedAt
    ? fmtDuration(Math.max(Number(session.uptime_seconds || 0), Math.floor(Date.now() / 1000 - startedAt)))
    : fmtDuration(session.uptime_seconds || 0);
  const findings = Array.isArray(latest.findings) ? latest.findings : [];
  const tensors = Object.entries(latest.tensors || {}).slice(0, 5);
  const hasRunStats = Number(step) > 0 || metricKeys.size > 0 || findings.length > 0 ||
    tensors.length > 0 || Number(session.uptime_seconds || 0) > 0;
  const canSend = Boolean(currentUser && (
    currentUser.id === session.user_id ||
    (currentTeam?.admin_ids || []).includes(currentUser.id)
  ));

  const entries = logs.map(entry => ({
    at: Number(entry.t || 0),
    html: `
      <article class="console-exchange">
        <div class="console-speaker">YOU <time>${fmtRelativeTime(entry.t || 0)}</time></div>
        <pre class="console-message is-user">${escapeHtml(entry.question || "")}</pre>
        <div class="console-speaker">PULSE</div>
        <pre class="console-message">${escapeHtml(entry.answer || "")}</pre>
      </article>
    `
  }));

  commands.filter(command => !logs.some(entry => entry.question === command.command)).forEach(command => {
    entries.push({
      at: command.created_at ? Date.parse(command.created_at) / 1000 : 0,
      html: `
        <article class="console-exchange">
          <div class="console-speaker">YOU
            <time>${fmtRelativeTime(command.created_at ? Date.parse(command.created_at) / 1000 : 0)}</time>
            <span class="command-state is-${escapeHtml(command.status || "pending")}">${escapeHtml(command.status || "pending")}</span>
          </div>
          <pre class="console-message is-user">${escapeHtml(command.command || "")}</pre>
          ${command.result ? `<pre class="console-message">${escapeHtml(command.result)}</pre>` : ""}
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
      <div class="run-side-stat"><span>STEP</span><strong>${escapeHtml(metricValue(step))}</strong></div>
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
        <section class="run-transcript" aria-label="Agent transcript">${transcript}</section>
      </div>
      <form class="command-composer" data-run-id="${escapeHtml(session.id)}">
        <label class="sr-only" for="command-${escapeHtml(session.id)}">Send a prompt to this run</label>
        <div class="command-input-wrap">
        <div class="command-suggestions" role="listbox" aria-label="Pulse commands" hidden></div>
        <textarea class="command-field" id="command-${escapeHtml(session.id)}" name="command" rows="2" maxlength="8000"
          placeholder="Ask about this run or enter a /command…" ${canSend ? "" : "disabled"} required></textarea>
        </div>
        <div class="command-composer-foot">
          <span class="command-status" role="status">${canSend ? "Enter to send · commands execute on the runner" : "Only the run owner or workspace admins can send commands"}</span>
          <button type="submit" title="Send command" ${canSend ? "" : "disabled"}>Send <span aria-hidden="true">↗</span></button>
        </div>
      </form>
    </section>
  `;
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

  if (
    !currentProject ||
    refreshInFlight
  ) {
    return;
  }

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

    // Switched project while this was loading.
    if (project !== currentProject) {
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

    console.error(error);

    els.refreshLabel.textContent =
      "Update failed";

  } finally {

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
  if (!event.target.matches(".command-field")) return;
  commandDrafts.set(activeSessionId, event.target.value);
  commandSuggestionIndex = 0;
  updateCommandSuggestions();
});


els.activeConsole.addEventListener("click", event => {
  const suggestion = event.target.closest(".command-suggestion");
  if (suggestion) chooseCommandSuggestion(suggestion.dataset.command);
});


els.activeConsole.addEventListener("keydown", event => {
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
    event.target.form.requestSubmit();
  }
});


els.activeConsole.addEventListener("submit", async event => {

  const form = event.target.closest(".command-composer");
  if (!form) return;
  event.preventDefault();

  const session = currentSessions.find(item => item.id === form.dataset.runId);
  const field = form.elements.command;
  const status = form.querySelector(".command-status");
  const button = form.querySelector("button[type=submit]");
  const command = field.value.trim();
  if (!session || !command || !currentUser) return;

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

if (els.projectsNotificationsBtn) {

  els.projectsNotificationsBtn.addEventListener(
    "click",
    openNotificationsModal
  );
}

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


setInterval(
  () => {

    if (
      !els.viewProjects.hidden &&
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
  if (script) return String(script).split(/[\\/]/).pop();
  return `run ${String(session.id || "").slice(0, 8)}`;
}


function renderRunPickerItem(session) {
  const live = isLive(session);
  const loss = findMetric(session, ["loss", "train_loss", "loss_value", "current_loss"]);
  const step = findMetric(session, ["step", "global_step", "epoch"]);
  const selected = session.id === activeSessionId;
  return `
    <button class="run-picker-item ${selected ? "is-selected" : ""}" type="button"
      data-session-id="${escapeHtml(session.id)}" aria-pressed="${selected}">
      <span class="run-picker-dot ${live ? "is-live" : ""}" aria-hidden="true"></span>
      <span class="run-picker-copy">
        <strong>${escapeHtml(runLabel(session))}</strong>
        <span>step ${escapeHtml(metricValue(step))} <i>·</i> loss ${escapeHtml(metricValue(loss))}</span>
      </span>
      <span class="run-picker-arrow" aria-hidden="true">›</span>
    </button>
  `;
}


function renderRunGroup(label, sessions) {
  if (!sessions.length) return "";
  return `
    <section class="run-group">
      <h2><span>${label}</span><b>${sessions.length}</b></h2>
      <div class="run-group-items">${sessions.map(renderRunPickerItem).join("")}</div>
    </section>
  `;
}


function renderSessions() {

  const liveRuns = currentSessions.filter(isLive);
  const stoppedRuns = currentSessions.filter(session => !isLive(session));
  if (!currentSessions.some(session => session.id === activeSessionId)) {
    activeSessionId = (liveRuns[0] || stoppedRuns[0])?.id || null;
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
      session.commands?.map(command => [command.id, command.status, command.result]),
      isLive(session)
    ])
  ]);
  if (signature === lastRenderedSignature) return;
  lastRenderedSignature = signature;

  renderStats(currentSessions);
  els.emptyState.hidden = currentSessions.length > 0;
  els.emptyState.textContent = "No runs have been logged for this project yet.";
  const sidebarScrollTop = els.sessionList.scrollTop;
  els.sessionList.innerHTML = [
    renderRunGroup("LIVE", liveRuns),
    renderRunGroup("STOPPED", stoppedRuns)
  ].join("");
  els.sessionList.scrollTop = sidebarScrollTop;

  const session = currentSessions.find(item => item.id === activeSessionId);
  const oldField = els.activeConsole.querySelector(".command-field");
  const wasFocused = oldField && document.activeElement === oldField;
  const oldTranscript = els.activeConsole.querySelector(".run-transcript");
  const oldTranscriptScroll = oldTranscript?.scrollTop || 0;
  const transcriptAtBottom = !oldTranscript ||
    oldTranscript.scrollHeight - oldTranscript.clientHeight - oldTranscript.scrollTop < 32;
  const draft = commandDrafts.get(activeSessionId) || "";
  els.activeConsole.innerHTML = session
    ? renderRunWorkspace(session)
    : `<div class="console-empty-state"><span>PULSE / DEBUG</span><p>Select a run to open its console.</p></div>`;

  const transcript = els.activeConsole.querySelector(".run-transcript");
  if (transcript) {
    transcript.scrollTop = transcriptAtBottom ? transcript.scrollHeight : oldTranscriptScroll;
  }

  const field = els.activeConsole.querySelector(".command-field");
  if (field) {
    field.value = draft;
    if (wasFocused) field.focus();
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