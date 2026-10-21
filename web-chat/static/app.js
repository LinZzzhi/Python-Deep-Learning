'use strict';

/* SimpleChat 前端
   与后端的通信全部是普通 HTTP 请求（1 秒轮询），不依赖 WebSocket / SSE。 */

const $ = (id) => document.getElementById(id);
const POLL_MS = 1000;
const LOBBY_POLL_MS = 2500;
const ADMIN_POLL_MS = 5000;
const ADMIN_NAME = 'admin';       // 昵称填这个就弹管理员密码框

const state = {
  session: null,      // {roomId, token, ghost}
  lastMsgId: 0,
  pollTimer: null,
  pollBusy: false,    // 是否有 state 请求正在飞行（保证永不并发）
  pollAgain: false,   // 忙碌期间有人请求刷新，回来后补一轮
  lobbyTimer: null,
  searchTimer: null,
  rendered: new Set(), // 已画到面板上的消息 id，防止同一条被重复渲染
  admin: null,        // {token, user}：管理员登录态
  adminTimer: null,
  createBusy: false,  // 创建按钮在途锁（防连击建出僵尸房）
  joinBusy: false,    // 加入按钮在途锁
};

/* ------------------------------------------------------------ 基础工具 */

async function api(path, options) {
  const res = await fetch(path, options || {});
  const text = await res.text();
  try {
    return JSON.parse(text);
  } catch (e) {
    throw new Error('服务器返回异常');
  }
}

function post(path, body) {
  return api(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  });
}

async function safe(fn) {
  try {
    return await fn();
  } catch (e) {
    toast('网络异常，请稍后重试');
    return null;
  }
}

let toastTimer = null;
function toast(message) {
  const el = $('toast');
  el.textContent = message;
  el.classList.remove('hidden');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.add('hidden'), 2800);
}

function nickname() {
  const value = $('nickname').value.trim();
  if (!value) {
    toast('请先填写你的昵称');
    $('nickname').focus();
    return '';
  }
  return value;
}

/* 在途锁配套：请求期间把按钮禁掉，避免连击发出多个一模一样的请求 */
function setBusy(btn, busy, label) {
  if (!btn) return;
  btn.disabled = !!busy;
  if (label) btn.textContent = label;
  btn.style.opacity = busy ? '.6' : '';
  btn.style.cursor = busy ? 'progress' : '';
}

/* 记住自己是哪些房间的创建者：关掉浏览器再回来权限不丢 */
function ownerKey(roomId) {
  try { return localStorage.getItem('sc-own-' + roomId) || ''; }
  catch (e) { return ''; }
}

function rememberOwner(roomId, token) {
  if (!roomId || !token) return;
  try { localStorage.setItem('sc-own-' + roomId, token); } catch (e) { /* ignore */ }
}

function fmtTime(ts) {
  const d = new Date(ts * 1000);
  const p = (n) => String(n).padStart(2, '0');
  return p(d.getHours()) + ':' + p(d.getMinutes());
}

/* ------------------------------------------------------------ 密码弹窗 */

/* 通用弹窗：默认是 6 位数字的房间密码；opts.mask 时变成用圆点遮挡的普通密码框 */
function askPassword(title, desc, preset, onOk, opts) {
  opts = opts || {};
  const modal = $('modal');
  const input = $('modal-input');
  $('modal-title').textContent = title;
  $('modal-desc').textContent = desc;
  input.type = opts.mask ? 'password' : 'text';
  input.className = opts.mask ? 'secret' : 'code';
  input.maxLength = opts.maxLength || 6;
  input.inputMode = opts.mask ? 'text' : 'numeric';
  input.placeholder = opts.placeholder || '';
  $('modal-ok').textContent = opts.okText || '加入';
  input.value = preset || '';
  modal.classList.remove('hidden');
  input.focus();
  if (opts.mask) input.select();

  const close = () => {
    modal.classList.add('hidden');
    $('modal-ok').onclick = null;
    $('modal-cancel').onclick = null;
    input.onkeydown = null;
  };
  const submit = () => {
    const value = input.value.trim();
    close();
    onOk(value);
  };
  $('modal-cancel').onclick = close;
  $('modal-ok').onclick = submit;
  input.onkeydown = (e) => {
    if (e.key === 'Enter') { e.preventDefault(); submit(); }
  };
}

/* ------------------------------------------------------------ 大厅 */

async function refreshRooms() {
  const q = $('search').value.trim();
  const data = await safe(() => api('/api/rooms?q=' + encodeURIComponent(q)));
  if (!data || !data.ok) return;
  renderRooms(data.rooms);
}

function renderRooms(rooms) {
  const list = $('room-list');
  list.textContent = '';
  $('room-total').textContent = rooms.length + ' 个';
  $('room-empty').classList.toggle('hidden', rooms.length > 0);

  rooms.forEach((room) => {
    const li = document.createElement('li');
    li.className = 'room-item';

    const info = document.createElement('div');
    const name = document.createElement('div');
    name.className = 'name';
    const nameText = document.createElement('span');
    nameText.textContent = room.name;
    name.appendChild(nameText);

    const tag = document.createElement('span');
    if (room.has_password) {
      tag.className = 'tag lock';
      tag.textContent = '密码';
    } else {
      tag.className = 'tag open';
      tag.textContent = '公开';
    }
    name.appendChild(tag);

    const meta = document.createElement('div');
    meta.className = 'meta';
    meta.textContent = '在线 ' + room.count + ' / ' + room.max + ' 人';

    info.appendChild(name);
    info.appendChild(meta);

    const btn = document.createElement('button');
    btn.textContent = '加入';
    btn.onclick = () => joinRoom(room, '');

    li.appendChild(info);
    li.appendChild(btn);
    list.appendChild(li);
  });
}

function startLobbyPolling() {
  stopLobbyPolling();
  const tick = async () => {
    if (!$('lobby').classList.contains('hidden')) await refreshRooms();
    state.lobbyTimer = setTimeout(tick, LOBBY_POLL_MS);
  };
  state.lobbyTimer = setTimeout(tick, LOBBY_POLL_MS);
}

function stopLobbyPolling() {
  clearTimeout(state.lobbyTimer);
  state.lobbyTimer = null;
}

/* ------------------------------------------------------------ 创建 / 加入 */

/* room 为 null 表示直接凭密码加入 */
async function createRoom() {
  if (state.createBusy) return;                 // 连击保护：请求没回来之前不再发第二次
  if (isAdminName()) { promptAdminLogin(); return; }
  const user = nickname();
  if (!user) return;
  const name = $('new-room').value.trim();
  if (!name) {
    toast('请填写聊天室名称');
    $('new-room').focus();
    return;
  }
  state.createBusy = true;
  const btn = $('create-btn');
  setBusy(btn, true, '创建中…');
  try {
    const data = await safe(() => post('/api/rooms', {
      name: name,
      user: user,
      password: $('use-password').checked,
    }));
    if (!data) return;
    if (!data.ok) { toast(data.error || '创建失败'); return; }
    $('new-room').value = '';
    if (data.is_owner) rememberOwner(data.room_id, data.token);
    enterRoom(data);
    if (data.password) toast('聊天室已创建，进房密码：' + data.password);
  } finally {
    state.createBusy = false;
    setBusy(btn, false, '创建并进入');
  }
}

async function joinRoom(room, password) {
  if (state.joinBusy) return;                   // 连击保护
  if (isAdminName()) { promptAdminLogin(); return; }
  const user = nickname();
  if (!user) return;

  if (room && room.has_password && !password) {
    askPassword('输入进房密码', '「' + room.name + '」需要密码才能加入', '',
                (value) => { if (value) joinRoom(room, value); });
    return;
  }

  state.joinBusy = true;
  const btn = $('join-btn');
  setBusy(btn, true, '加入中…');
  try {
    const data = await safe(() => post('/api/rooms/join', {
      room_id: room ? room.id : '',
      password: password || '',
      user: user,
      resume_token: room ? ownerKey(room.id) : '',
    }));
    if (!data) return;
    if (!data.ok) { toast(data.error || '加入失败'); return; }
    if (data.is_owner) rememberOwner(data.room_id, data.token);
    enterRoom(data);
  } finally {
    state.joinBusy = false;
    setBusy(btn, false, '加入');
  }
}

/* ------------------------------------------------------------ 房间 */

function enterRoom(data) {
  state.session = { roomId: data.room_id, token: data.token };
  state.lastMsgId = 0;
  state.rendered.clear();
  sessionStorage.setItem('simplechat', JSON.stringify(state.session));
  $('messages').textContent = '';
  showRoom();
  requestPoll(true);
}

/* 无痕会话（管理员）的标记放在 session 上，界面据此换提示与按钮文案 */
function isGhostSession() {
  return !!(state.session && state.session.ghost);
}

function updateGhostUi() {
  const ghost = isGhostSession();
  $('ghost-banner').classList.toggle('hidden', !ghost);
  $('leave-btn').textContent = ghost ? '退出无痕' : '离开';
}

function showRoom() {
  stopLobbyPolling();
  stopAdminPolling();
  $('lobby').classList.add('hidden');
  $('admin').classList.add('hidden');
  $('room').classList.remove('hidden');
  updateGhostUi();
  $('message-input').focus();
}

function showLobbyPage() {
  stopAdminPolling();
  $('room').classList.add('hidden');
  $('admin').classList.add('hidden');
  $('lobby').classList.remove('hidden');
  startLobbyPolling();
  refreshRooms();
}

/* 清掉房间会话相关的所有前端状态（离开房间、换房间、被踢出都走这里） */
function resetSessionState() {
  stopPolling();
  state.session = null;
  state.lastMsgId = 0;
  state.pollBusy = false;
  state.pollAgain = false;
  state.rendered.clear();
  sessionStorage.removeItem('simplechat');
  $('messages').textContent = '';
  $('member-list').textContent = '';
  updateGhostUi();
}

function exitToLobby(message) {
  resetSessionState();
  showLobbyPage();
  if (message) toast(message);
}

/* 无痕会话结束后回到管理员控制台（没有管理员登录态就退回大厅） */
function exitToAdmin(message) {
  const wasAdmin = !!state.admin;
  resetSessionState();
  if (wasAdmin) showAdmin();
  else showLobbyPage();
  if (message) toast(message);
}

/* 房间会话因为各种原因结束时，决定回到哪个界面 */
function leaveSession(message) {
  if (isGhostSession()) exitToAdmin(message);
  else exitToLobby(message);
}

function schedulePoll(delay) {
  clearTimeout(state.pollTimer);
  state.pollTimer = setTimeout(pollRoom, delay);
}

function stopPolling() {
  clearTimeout(state.pollTimer);
  state.pollTimer = null;
}

/* 请求刷新服务端状态。immediate=true 表示尽快拉一次（发完消息、移人之后）。
   关键点：任何时刻只允许一个 state 请求在飞。若已有请求在飞，只做标记，
   等它回来后立刻补一轮。否则两次请求会带着同一个 since，把同一批消息
   各画一遍 —— 这就是「自己的消息显示两遍」的根因。 */
function requestPoll(immediate) {
  if (state.pollBusy) {
    if (immediate) state.pollAgain = true;
    return;
  }
  if (immediate) {
    clearTimeout(state.pollTimer);
    state.pollTimer = null;
    pollRoom();
    return;
  }
  schedulePoll(0);
}

async function pollRoom() {
  if (!state.session) return;
  if (state.pollBusy) { state.pollAgain = true; return; }

  state.pollBusy = true;
  clearTimeout(state.pollTimer);
  state.pollTimer = null;
  // 以「上一轮开始时刻 + POLL_MS」为下一次的目标，抵消请求本身耗时造成的漂移
  const due = Date.now() + POLL_MS;
  const roomId = state.session.roomId;
  const token = state.session.token;
  const since = state.lastMsgId;
  const owner = state.session;      // 用来识别「响应回来时已经换了房间」的陈旧数据

  let next = 1200;          // 网络异常时的重试间隔
  let leave = null;
  let ok = false;
  try {
    const data = await safe(() => api(
      '/api/rooms/state?room_id=' + encodeURIComponent(roomId) +
      '&token=' + encodeURIComponent(token) +
      '&since=' + since));
    if (!data) {
      next = 1200;
    } else if (!data.ok) {
      leave = data.error || '你已离开该聊天室';
    } else {
      ok = true;
      renderRoom(data);
      next = Math.max(60, due - Date.now());
    }
  } finally {
    state.pollBusy = false;
  }

  if (state.session !== owner) return;   // 期间换了房间，这次响应作废

  if (leave) { leaveSession(leave); return; }
  if (ok && state.pollAgain) {
    state.pollAgain = false;
    schedulePoll(0);        // 期间刚发过消息，立刻补一轮
    return;
  }
  schedulePoll(next);
}

function messageNode(msg, myMid) {
  const li = document.createElement('li');
  if (msg.system) {
    li.className = 'msg system';
    const body = document.createElement('div');
    body.className = 'body';
    body.textContent = msg.text;
    li.appendChild(body);
    return li;
  }

  const mine = msg.mid && msg.mid === myMid;
  li.className = 'msg' + (mine ? ' mine' : '');

  const who = document.createElement('div');
  who.className = 'who';
  const name = document.createElement('span');
  if (mine) name.className = 'me';
  name.textContent = msg.user;
  const time = document.createElement('span');
  time.textContent = fmtTime(msg.ts);
  who.appendChild(name);
  who.appendChild(time);

  const body = document.createElement('div');
  body.className = 'body';
  body.textContent = msg.text;          // 用 textContent，避免 XSS

  li.appendChild(who);
  li.appendChild(body);
  return li;
}

function renderRoom(data) {
  $('room-title').textContent = data.room.name;
  $('room-count').textContent = data.room.count;
  $('member-count').textContent = data.room.count;

  const pw = $('room-password');
  if (data.password) {
    pw.textContent = '进房密码 ' + data.password;
    pw.title = '把这个密码发给要邀请的人';
    pw.classList.remove('hidden');
  } else {
    pw.classList.add('hidden');
  }

  const isOwner = !!data.you.is_owner;
  const isGhost = !!data.you.ghost;
  $('dissolve-btn').classList.toggle('hidden', !isOwner);
  $('owner-tip').classList.toggle('hidden', !isOwner);
  $('owner-tip').textContent = isGhost
    ? '无痕管理员：你拥有房主权限，可以移出成员或解散房间。'
    : '你是创建者，可以把成员移出聊天室。';
  $('ghost-banner').classList.toggle('hidden', !isGhost);
  $('admin-view-tip').classList.toggle('hidden', !data.admin_view);

  if (data.messages.length) {
    const box = $('messages');
    const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 80;
    let appended = 0;
    data.messages.forEach((m) => {
      if (m.id > state.lastMsgId) state.lastMsgId = m.id;
      if (state.rendered.has(m.id)) return;   // 双保险：同一条消息只画一次
      state.rendered.add(m.id);
      box.appendChild(messageNode(m, data.you.mid));
      appended++;
    });
    if (appended && atBottom) box.scrollTop = box.scrollHeight;
  }

  const list = $('member-list');
  list.textContent = '';
  data.members.forEach((member) => {
    const li = document.createElement('li');
    const nm = document.createElement('div');
    nm.className = 'nm';
    const name = document.createElement('span');
    name.textContent = member.name + (member.is_you ? '（我）' : '');
    nm.appendChild(name);
    if (member.ghost) {
      const tag = document.createElement('span');
      tag.className = 'tag ghost';
      tag.textContent = '无痕';
      nm.appendChild(tag);
    }
    if (member.is_owner) {
      const tag = document.createElement('span');
      tag.className = 'tag lock';
      tag.textContent = '房主';
      nm.appendChild(tag);
    }
    li.appendChild(nm);

    if (data.admin_view && member.ip) {
      const ip = document.createElement('span');
      ip.className = 'ip';
      ip.textContent = member.ip;
      ip.title = '成员 IP（仅管理员可见）';
      li.appendChild(ip);
    }

    if (isOwner && !member.is_you) {
      const btn = document.createElement('button');
      btn.className = 'mini danger';
      btn.textContent = '移出';
      btn.onclick = () => kickMember(member);
      li.appendChild(btn);
    }
    list.appendChild(li);
  });
}

/* ------------------------------------------------------------ 房间内操作 */

async function sendMessage(event) {
  event.preventDefault();
  if (!state.session) return;
  const input = $('message-input');
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  const data = await safe(() => post('/api/rooms/send', {
    room_id: state.session.roomId,
    token: state.session.token,
    text: text,
  }));
  if (!data) return;
  if (!data.ok) {
    toast(data.error || '发送失败');
    if (data.gone || data.removed) leaveSession(data.error);
    return;
  }
  requestPoll(true);                    // 立刻拉一次，自己的消息即时显示
}

async function kickMember(member) {
  if (!state.session) return;
  if (!window.confirm('确定把「' + member.name + '」移出聊天室吗？')) return;
  const data = await safe(() => post('/api/rooms/kick', {
    room_id: state.session.roomId,
    token: state.session.token,
    target: member.mid,
  }));
  if (!data) return;
  if (!data.ok) { toast(data.error || '操作失败'); return; }
  requestPoll(true);
}

async function dissolveRoom() {
  if (!state.session) return;
  if (!window.confirm('确定解散该聊天室吗？房间内所有人都会被移出，且无法恢复。')) return;
  const data = await safe(() => post('/api/rooms/dissolve', {
    room_id: state.session.roomId,
    token: state.session.token,
  }));
  if (!data) return;
  if (!data.ok) { toast(data.error || '操作失败'); return; }
  leaveSession('聊天室已解散');
}

function leavePayload() {
  if (!state.session) return null;
  return JSON.stringify({
    room_id: state.session.roomId,
    token: state.session.token,
  });
}

function leaveRoom() {
  const payload = leavePayload();
  const ghost = isGhostSession();
  if (payload && navigator.sendBeacon) {
    // 让服务端立刻把在线人数减掉，不用等 15 秒超时
    navigator.sendBeacon('/api/rooms/leave',
                         new Blob([payload], { type: 'application/json' }));
  } else if (payload) {
    post('/api/rooms/leave', JSON.parse(payload));
  }
  leaveSession(ghost ? '已退出无痕模式' : '已离开聊天室');
}

/* ------------------------------------------------------------ 管理员 */

function isAdminName() {
  return $('nickname').value.trim().toLowerCase() === ADMIN_NAME;
}

function loadAdminSession() {
  try {
    const raw = sessionStorage.getItem('simplechat-admin');
    if (!raw) return;
    const saved = JSON.parse(raw);
    if (saved && saved.token) state.admin = { token: saved.token, user: saved.user || ADMIN_NAME };
  } catch (e) { /* ignore */ }
}

function saveAdminSession() {
  try {
    if (state.admin) sessionStorage.setItem('simplechat-admin', JSON.stringify(state.admin));
    else sessionStorage.removeItem('simplechat-admin');
  } catch (e) { /* ignore */ }
}

/* 昵称是 admin 时弹密码框（内容必须自己输入，不做任何预填） */
function promptAdminLogin() {
  askPassword('管理员验证', '昵称是 admin，请输入管理员密码', '',
              (value) => {
                if (!value) { toast('请输入管理员密码'); return; }
                adminLogin(value);
              },
              { mask: true, maxLength: 64, okText: '进入', placeholder: '请输入管理员密码' });
}

async function adminLogin(password) {
  const user = $('nickname').value.trim() || ADMIN_NAME;
  const data = await safe(() => post('/api/admin/login', { user: user, password: password }));
  if (!data) return;
  if (!data.ok) { toast(data.error || '管理员登录失败'); return; }
  state.admin = { token: data.token, user: data.user };
  saveAdminSession();
  showAdmin();
  toast('已进入管理员控制台');
}

function showAdmin() {
  stopLobbyPolling();
  $('lobby').classList.add('hidden');
  $('room').classList.add('hidden');
  $('admin').classList.remove('hidden');
  $('admin-user').textContent = state.admin ? state.admin.user : ADMIN_NAME;
  adminFetchData();
  startAdminPolling();
}

function adminLogout() {
  stopAdminPolling();
  state.admin = null;
  saveAdminSession();
  showLobbyPage();
  toast('已退出管理员');
}

function adminExpired(data) {
  stopAdminPolling();
  state.admin = null;
  saveAdminSession();
  toast((data && data.error) || '管理员登录已过期，请重新登录');
  if (isGhostSession()) exitToLobby(null);
  else showLobbyPage();
}

function startAdminPolling() {
  stopAdminPolling();
  const tick = async () => {
    if (state.admin && !state.session && !$('admin').classList.contains('hidden')) {
      await adminFetchData();
    }
    state.adminTimer = setTimeout(tick, ADMIN_POLL_MS);
  };
  state.adminTimer = setTimeout(tick, ADMIN_POLL_MS);
}

function stopAdminPolling() {
  clearTimeout(state.adminTimer);
  state.adminTimer = null;
}

async function adminFetchData() {
  if (!state.admin) return;
  const data = await safe(() => api('/api/admin/data?token=' +
                                    encodeURIComponent(state.admin.token)));
  if (!data) return;
  if (!data.ok) { adminExpired(data); return; }
  state.admin.user = data.user;
  saveAdminSession();
  renderAdmin(data);
}

async function adminSaveAccount() {
  if (!state.admin) return;
  const btn = $('admin-save');
  setBusy(btn, true, '保存中…');
  try {
    const data = await safe(() => post('/api/admin/update', {
      token: state.admin.token,
      user: $('admin-name').value.trim(),
      password: $('admin-pass').value,
    }));
    if (!data) return;
    if (!data.ok) { adminExpiredIfNeeded(data); return; }
    state.admin.user = data.user;
    saveAdminSession();
    $('admin-user').textContent = data.user;
    toast('管理员账号已更新');
  } finally {
    setBusy(btn, false, '保存修改');
  }
}

function adminExpiredIfNeeded(data) {
  if (data && data.admin_expired) adminExpired(data);
  else toast(data.error || '操作失败');
}

function toggleAdminPassword() {
  const input = $('admin-pass');
  const shown = input.type === 'text';
  input.type = shown ? 'password' : 'text';
  $('admin-eye').textContent = shown ? '显示' : '隐藏';
}

/* 无痕加入：不进成员列表、不计人数、不发提示，但拥有房主权限 */
async function adminGhostJoin(roomId) {
  if (!state.admin) return;
  const data = await safe(() => post('/api/admin/ghost', {
    token: state.admin.token,
    room_id: roomId,
  }));
  if (!data) return;
  if (!data.ok) { adminExpiredIfNeeded(data); return; }
  stopAdminPolling();
  state.session = { roomId: data.room_id, token: data.token, ghost: true };
  state.lastMsgId = 0;
  state.pollBusy = false;
  state.pollAgain = false;
  state.rendered.clear();
  sessionStorage.setItem('simplechat', JSON.stringify(state.session));
  $('messages').textContent = '';
  $('member-list').textContent = '';
  showRoom();
  requestPoll(true);
  toast('已无痕进入「' + (data.room && data.room.name || '聊天室') + '」');
}

function fmtFull(ts) {
  if (!ts) return '';
  const d = new Date(ts * 1000);
  const p = (n) => String(n).padStart(2, '0');
  return p(d.getMonth() + 1) + '-' + p(d.getDate()) + ' ' +
         p(d.getHours()) + ':' + p(d.getMinutes()) + ':' + p(d.getSeconds());
}

function cell(text, cls) {
  const td = document.createElement('td');
  td.textContent = text === null || text === undefined ? '' : String(text);
  if (cls) td.className = cls;
  return td;
}

function adminTable(headers, rows) {
  const table = document.createElement('table');
  table.className = 'adm-table';
  const thead = document.createElement('thead');
  const tr = document.createElement('tr');
  headers.forEach((h) => {
    const th = document.createElement('th');
    th.textContent = h;
    tr.appendChild(th);
  });
  thead.appendChild(tr);
  table.appendChild(thead);
  const tbody = document.createElement('tbody');
  rows.forEach((cells) => {
    const row = document.createElement('tr');
    cells.forEach((c) => row.appendChild(c));
    tbody.appendChild(row);
  });
  table.appendChild(tbody);
  return table;
}

function renderAdmin(data) {
  $('admin-user').textContent = data.user;
  const nameInput = $('admin-name');
  const passInput = $('admin-pass');
  // 正在输入时不要被轮询覆盖
  if (document.activeElement !== nameInput) nameInput.value = data.user;
  if (document.activeElement !== passInput) passInput.value = data.password || '';

  const stats = data.stats || {};
  $('admin-stats').textContent = '房间 ' + (stats.rooms || 0) + ' · 进行中 ' +
    (stats.active || 0) + ' · 消息 ' + (stats.messages || 0) +
    ' · 用户 ' + (stats.users || 0);

  const users = $('admin-userlist');
  users.textContent = '';
  (data.users || []).forEach((u) => {
    const li = document.createElement('li');
    const nm = document.createElement('span');
    nm.textContent = u.name;
    const ip = document.createElement('span');
    ip.className = 'ip';
    ip.textContent = u.ip || '-';
    li.appendChild(nm);
    li.appendChild(ip);
    users.appendChild(li);
  });
  if (!(data.users || []).length) {
    const li = document.createElement('li');
    li.textContent = '暂无数据';
    users.appendChild(li);
  }

  const box = $('admin-rooms');
  box.textContent = '';
  $('admin-room-count').textContent = (data.rooms || []).length + ' 个';
  if (!(data.rooms || []).length) {
    const p = document.createElement('p');
    p.className = 'empty';
    p.textContent = '还没有任何聊天室记录';
    box.appendChild(p);
    return;
  }

  data.rooms.forEach((room) => box.appendChild(adminRoomNode(room)));
}

function adminRoomNode(room) {
  const wrap = document.createElement('div');
  wrap.className = 'adm-room';

  const head = document.createElement('div');
  head.className = 'adm-room-head';

  const info = document.createElement('div');
  const name = document.createElement('div');
  name.className = 'name';
  const nameText = document.createElement('span');
  nameText.textContent = room.name;
  name.appendChild(nameText);

  const stateTag = document.createElement('span');
  stateTag.className = 'tag ' + (room.active ? 'live' : 'dead');
  stateTag.textContent = room.active ? '进行中' : '已结束';
  name.appendChild(stateTag);

  if (room.has_password) {
    const tag = document.createElement('span');
    tag.className = 'tag lock';
    tag.textContent = '密码 ' + (room.password || '');
    name.appendChild(tag);
  }
  info.appendChild(name);

  const meta = document.createElement('div');
  meta.className = 'meta';
  meta.textContent = '创建者 ' + (room.owner || '?') +
    ' · 创建于 ' + fmtFull(room.created) +
    (room.ended ? ' · 结束于 ' + fmtFull(room.ended) : '') +
    ' · 当前在线 ' + (room.online || 0) +
    ' · 共 ' + (room.participants || []).length + ' 位访客' +
    ' · ' + (room.messages || []).length + ' 条消息' +
    ' · 房间号 ' + room.id;
  info.appendChild(meta);
  head.appendChild(info);

  if (room.active) {
    const btn = document.createElement('button');
    btn.className = 'mini primary';
    btn.textContent = '无痕加入';
    btn.onclick = () => adminGhostJoin(room.id);
    head.appendChild(btn);
  }
  wrap.appendChild(head);

  const details = document.createElement('details');
  const summary = document.createElement('summary');
  summary.textContent = '查看参与者 IP 与全部消息';
  details.appendChild(summary);

  const cols = document.createElement('div');
  cols.className = 'adm-cols';

  const left = document.createElement('div');
  const h1 = document.createElement('h4');
  h1.textContent = '参与者（用户名 / IP）';
  left.appendChild(h1);
  const scrollL = document.createElement('div');
  scrollL.className = 'adm-scroll';
  scrollL.appendChild(adminTable(
    ['用户名', 'IP', '加入', '离开'],
    (room.participants || []).map((p) => [
      cell(p.name + (p.ghost ? '（无痕）' : '')),
      cell(p.ip || '-', 'ip'),
      cell(fmtFull(p.joined), 'dim'),
      cell(p.left ? fmtFull(p.left) : '在线', 'dim'),
    ])));
  left.appendChild(scrollL);

  const right = document.createElement('div');
  const h2 = document.createElement('h4');
  h2.textContent = '消息内容';
  right.appendChild(h2);
  const scrollR = document.createElement('div');
  scrollR.className = 'adm-scroll';
  scrollR.appendChild(adminTable(
    ['时间', '用户', '内容', 'IP'],
    (room.messages || []).map((m) => [
      cell(fmtFull(m.ts), 'dim'),
      cell(m.system ? '系统' : (m.user || ''), 'dim'),
      cell(m.text, 'txt'),
      cell(m.ip || '-', 'ip'),
    ])));
  right.appendChild(scrollR);

  cols.appendChild(left);
  cols.appendChild(right);
  details.appendChild(cols);
  wrap.appendChild(details);
  return wrap;
}

/* ------------------------------------------------------------ 启动 */

function resumeSession() {
  let raw = null;
  try {
    raw = sessionStorage.getItem('simplechat');
  } catch (e) { return; }
  if (!raw) return;
  let saved;
  try { saved = JSON.parse(raw); } catch (e) { saved = null; }
  if (!saved || !saved.roomId || !saved.token) return;
  state.session = { roomId: saved.roomId, token: saved.token, ghost: !!saved.ghost };
  state.lastMsgId = 0;
  state.pollBusy = false;
  state.pollAgain = false;
  state.rendered.clear();
  showRoom();
  requestPoll(true);
}

function init() {
  try {
    $('nickname').value = localStorage.getItem('simplechat-nick') || '';
  } catch (e) { /* 隐私模式忽略 */ }
  $('nickname').addEventListener('input', () => {
    try { localStorage.setItem('simplechat-nick', $('nickname').value.trim()); }
    catch (e) { /* ignore */ }
  });

  $('search').addEventListener('input', () => {
    clearTimeout(state.searchTimer);
    state.searchTimer = setTimeout(refreshRooms, 180);
  });
  $('create-btn').onclick = createRoom;
  $('join-btn').onclick = () => {
    const code = $('join-password').value.trim();
    if (!/^\d{6}$/.test(code)) { toast('请输入 6 位数字密码'); return; }
    joinRoom(null, code);
  };
  $('join-password').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); $('join-btn').click(); }
  });
  $('send-form').addEventListener('submit', sendMessage);
  $('leave-btn').onclick = leaveRoom;
  $('dissolve-btn').onclick = dissolveRoom;

  $('admin-refresh').onclick = adminFetchData;
  $('admin-lobby').onclick = () => { showLobbyPage(); };
  $('admin-logout').onclick = adminLogout;
  $('admin-save').onclick = adminSaveAccount;
  $('admin-eye').onclick = toggleAdminPassword;

  /* 注意：这里**不能**在 beforeunload 里主动调 leave 接口。
     否则用户一刷新就先把自己从房间里删掉，重新加载后只能看到「你已不在该聊天室内」，
     直接被踢回大厅。在线状态交给服务端的轮询心跳（15 秒超时）自然收敛即可。 */

  startLobbyPolling();
  refreshRooms();
  loadAdminSession();
  resumeSession();
}

document.addEventListener('DOMContentLoaded', init);
