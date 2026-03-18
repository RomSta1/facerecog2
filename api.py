import base64
import logging
import os
import subprocess
import threading

import cv2
import numpy as np
import yaml
from flask import Flask, jsonify, request, send_file, Response

log = logging.getLogger(__name__)

SNAPSHOT_DIR = "/opt/facerecog2/data/snapshots"


def _parse_snap_filename(fname):
    """cam_65_roman_143022.jpg → (cam_65, roman, 143022)"""
    base = fname[:-4]  # strip .jpg
    # last part is time (6 digits), second-last is name, rest is camera
    parts = base.split("_")
    if len(parts) >= 3:
        time_str = parts[-1]
        name = parts[-2]
        camera = "_".join(parts[:-2])
        return camera, name, time_str
    return base, "?", ""


HTML = r"""<!DOCTYPE html>
<html lang="uk">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FaceRecog2</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh}
header{background:#1e293b;padding:12px 20px;display:flex;align-items:center;gap:10px;border-bottom:1px solid #334155;position:sticky;top:0;z-index:10;flex-wrap:wrap}
header h1{font-size:1.1rem;font-weight:700}
.dot{width:9px;height:9px;border-radius:50%;background:#22c55e;flex-shrink:0}
.dot.off{background:#ef4444}
main{max-width:1200px;margin:0 auto;padding:16px;display:grid;gap:16px}
.card{background:#1e293b;border-radius:12px;padding:16px;border:1px solid #334155}
.card h2{font-size:.75rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:.06em;margin-bottom:12px}
label.lbl{font-size:.78rem;color:#94a3b8;display:block;margin-bottom:4px}
input[type=text],select{width:100%;background:#0f172a;border:1px solid #334155;color:#e2e8f0;padding:8px 10px;border-radius:8px;font-size:.88rem;outline:none}
input[type=text]:focus,select:focus{border-color:#6366f1}
.btn{padding:8px 16px;border-radius:8px;border:none;font-size:.85rem;font-weight:600;cursor:pointer;white-space:nowrap}
.bp{background:#6366f1;color:#fff}.bp:hover{background:#4f46e5}
.bg{background:#22c55e;color:#000}.bg:hover{background:#16a34a}
.br{background:#450a0a;color:#fca5a5}.br:hover{background:#7f1d1d}
.bs{padding:5px 10px;font-size:.78rem}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:10px}
.empty{color:#475569;font-size:.83rem;padding:6px 0}
/* snaps */
.sg{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px;max-height:500px;overflow-y:auto}
.sc{background:#0f172a;border-radius:8px;overflow:hidden;border:1px solid #334155}
.sc:hover{border-color:#6366f1}
.sc img{width:100%;height:110px;object-fit:cover;display:block;cursor:pointer}
.si{padding:6px 8px;font-size:.72rem}
.sn{font-weight:600;color:#f1f5f9;font-size:.8rem}
.sk{color:#64748b;margin-top:2px}
.sact{display:flex;gap:4px;margin-top:6px;align-items:center}
.sact select{flex:1;padding:4px 6px;font-size:.72rem;background:#1e293b;border:1px solid #475569;color:#e2e8f0;border-radius:6px}
.sact button{padding:4px 8px;border-radius:6px;border:none;font-size:.72rem;cursor:pointer;font-weight:600}
/* unknown */
.ug{display:grid;grid-template-columns:repeat(auto-fill,minmax(145px,1fr));gap:8px;max-height:360px;overflow-y:auto}
.uc{background:#0f172a;border-radius:8px;overflow:hidden;border:1px solid #334155}
.uc img{width:100%;height:100px;object-fit:cover;display:block;cursor:pointer}
.ui{padding:6px 8px}
.uk{color:#64748b;font-size:.68rem;margin-bottom:5px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.ua{display:flex;gap:4px;align-items:center}
.ua select{flex:1;padding:4px 6px;font-size:.72rem;background:#1e293b;border:1px solid #475569;color:#e2e8f0;border-radius:6px}
.ua button{padding:4px 7px;border-radius:6px;border:none;font-size:.72rem;cursor:pointer;font-weight:600;white-space:nowrap}
.ub{background:#6366f1;color:#fff}.ub:hover{background:#4f46e5}
.ud{background:#334155;color:#ef4444}.ud:hover{background:#450a0a}
/* persons columns */
.cols-wrap{display:flex;gap:10px;overflow-x:auto;align-items:flex-start;padding-bottom:4px}
.col-box{background:#0f172a;border-radius:10px;border:1px solid #334155;min-width:160px;flex:1}
.col-hdr{display:flex;align-items:center;gap:6px;padding:8px 10px;border-bottom:1px solid #334155}
.col-hdr .col-name{font-weight:700;font-size:.85rem;flex:1;cursor:default}
.col-hdr .col-name[contenteditable=true]{cursor:text;outline:1px solid #6366f1;border-radius:4px;padding:1px 4px}
.col-hdr button{background:none;border:none;cursor:pointer;font-size:.8rem;padding:2px 4px;border-radius:4px;color:#64748b}
.col-hdr button:hover{background:#1e293b;color:#e2e8f0}
.col-persons{padding:6px}
.pcard{display:flex;align-items:center;gap:6px;padding:6px 7px;border-radius:7px;cursor:pointer;border:1px solid transparent}
.pcard:hover{background:#1e293b;border-color:#334155}
.pcard .pav{font-size:1.3rem;flex-shrink:0}
.pcard .pinfo{flex:1;min-width:0}
.pcard .pname{font-weight:600;font-size:.83rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.pcard .pcnt{font-size:.7rem;color:#64748b}
.pcard .pdel{background:none;border:none;color:#ef4444;cursor:pointer;font-size:.78rem;opacity:0;padding:2px 4px;border-radius:4px}
.pcard:hover .pdel{opacity:.6}
.pcard:hover .pdel:hover{opacity:1}
.col-add{padding:8px;display:flex;justify-content:center}
.col-add-btn{background:#0f172a;border:1px dashed #334155;border-radius:10px;min-width:120px;padding:12px;text-align:center;cursor:pointer;color:#475569;font-size:.82rem}
.col-add-btn:hover{border-color:#6366f1;color:#a5b4fc}
.unassigned-hdr{font-size:.72rem;color:#475569;margin:8px 0 4px;font-weight:600;text-transform:uppercase;letter-spacing:.04em}
/* modals */
.mo{display:none;position:fixed;inset:0;background:rgba(0,0,0,.85);z-index:50;align-items:center;justify-content:center}
.mo.on{display:flex}
.mo>img{max-width:92vw;max-height:92vh;border-radius:6px}
.mc{position:fixed;top:12px;right:16px;background:none;border:none;color:#fff;font-size:2rem;cursor:pointer;line-height:1}
.mdlg{background:#1e293b;border-radius:12px;padding:22px;min-width:280px;max-width:94vw;max-height:90vh;overflow-y:auto}
.mdlg h3{font-size:.95rem;font-weight:700;margin-bottom:14px}
/* person photos */
.pm{display:none;position:fixed;inset:0;background:rgba(0,0,0,.92);z-index:60;flex-direction:column}
.pm.on{display:flex}
.pmh{background:#1e293b;padding:12px 18px;display:flex;align-items:center;gap:10px;border-bottom:1px solid #334155}
.pmh h3{font-size:.95rem;font-weight:700;flex:1}
.phg{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:8px;padding:14px;overflow-y:auto;flex:1}
.ph{background:#0f172a;border-radius:8px;overflow:hidden;border:1px solid #334155}
.ph img{width:100%;height:120px;object-fit:cover;display:block;cursor:pointer}
.ph .pname{padding:3px 6px;font-size:.68rem;color:#94a3b8;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
/* photo preview */
.pp{display:none;position:fixed;inset:0;background:rgba(0,0,0,.92);z-index:70;align-items:center;justify-content:center}
.pp.on{display:flex}
.pp img{max-width:92vw;max-height:88vh;border-radius:6px;object-fit:contain}
.pp .ppdel{position:fixed;top:14px;right:56px;background:rgba(239,68,68,.9);border:none;color:#fff;border-radius:50%;width:32px;height:32px;font-size:1rem;cursor:pointer;display:flex;align-items:center;justify-content:center}
.pp .ppdel:hover{background:#dc2626}
.pp .ppclose{position:fixed;top:10px;right:14px;background:none;border:none;color:#fff;font-size:2rem;cursor:pointer;line-height:1}
/* logs modal */
.lb{background:#020617;border-radius:8px;padding:12px;font-family:monospace;font-size:.7rem;color:#94a3b8;height:60vh;overflow-y:auto;white-space:pre-wrap;border:1px solid #1e293b;line-height:1.5;margin-top:10px}
/* settings */
.cfg-section{margin-bottom:18px}
.cfg-section h4{font-size:.72rem;font-weight:700;color:#6366f1;text-transform:uppercase;letter-spacing:.07em;margin-bottom:10px;padding-bottom:5px;border-bottom:1px solid #334155}
.cfg-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.cfg-field{display:flex;flex-direction:column;gap:4px}
.cfg-field label{font-size:.72rem;color:#94a3b8}
.cfg-field input,.cfg-field select{padding:6px 8px;border-radius:7px;border:1px solid #334155;background:#0f172a;color:#e2e8f0;font-size:.83rem}
.cfg-tabs{display:flex;gap:6px;margin-bottom:12px;flex-wrap:wrap}
.cfg-tab{padding:5px 12px;border-radius:7px;border:1px solid #334155;background:#0f172a;color:#94a3b8;font-size:.78rem;cursor:pointer}
.cfg-tab.active{background:#6366f1;color:#fff;border-color:#6366f1}
.cfg-cam{display:none}.cfg-cam.active{display:block}
</style>
</head>
<body>
<header>
  <div class="dot" id="dot"></div>
  <h1>FaceRecog2</h1>
  <span id="st" style="font-size:.78rem;color:#64748b">...</span>
  <div style="flex:1"></div>
  <label style="font-size:.78rem;color:#94a3b8">Камера:</label>
  <select id="camSel" onchange="onCamChange()" style="background:#0f172a;border:1px solid #334155;color:#e2e8f0;padding:5px 8px;border-radius:8px;font-size:.83rem;width:auto">
  </select>
  <button class="btn bp bs" onclick="reloadDB()">↺ База</button>
  <button class="btn bs" onclick="openLogs()" style="background:#334155;color:#e2e8f0">📋 Логи</button>
  <button class="btn bs" onclick="openSettings()" style="background:#334155;color:#e2e8f0">⚙️ Налаштування</button>
</header>
<main>

<!-- 1. ОСТАННІ ЗНІМКИ -->
<div class="card">
  <h2>Останні знімки</h2>
  <div class="row">
    <select id="sf" onchange="loadSnaps()" style="background:#0f172a;border:1px solid #334155;color:#e2e8f0;padding:5px 8px;border-radius:8px;font-size:.83rem;width:auto">
      <option value="">Всі особи</option>
    </select>
    <button class="btn bp bs" onclick="loadSnaps()">Оновити</button>
    <button class="btn bs" id="ab" onclick="toggleAuto()" style="background:#334155;color:#e2e8f0">▶ Авто</button>
    <button class="btn br bs" onclick="deleteAllSnaps()" style="margin-left:auto">🗑 Всі</button>
  </div>
  <div class="sg" id="sg"><div class="empty">Немає знімків</div></div>
</div>

<!-- 2. НЕВІДОМІ ОБЛИЧЧЯ -->
<div class="card">
  <h2>Невідомі обличчя</h2>
  <div class="row">
    <button class="btn bp bs" onclick="loadUnknown()">Оновити</button>
    <span id="ucnt" style="font-size:.76rem;color:#64748b"></span>
  </div>
  <div class="ug" id="ug"><div class="empty">Немає невідомих облич</div></div>
</div>

<!-- 3. ОСОБИ В БАЗІ — стовпці -->
<div class="card">
  <h2>Особи в базі</h2>
  <div class="cols-wrap" id="colsWrap"></div>
</div>

</main>

<!-- Image zoom modal -->
<div class="mo" id="mo" onclick="closeMo()">
  <button class="mc" onclick="closeMo()">✕</button>
  <img id="mi" src="">
</div>

<!-- Logs modal -->
<div class="mo" id="logsMo" onclick="closeLogsMo()" style="z-index:55;align-items:flex-start;padding:20px">
  <div class="mdlg" style="width:min(800px,96vw)" onclick="event.stopPropagation()">
    <div style="display:flex;align-items:center;margin-bottom:8px">
      <h3 style="flex:1">📋 Логи сервісу</h3>
      <button class="btn bp bs" onclick="loadLogs()" style="margin-right:8px">Оновити</button>
      <button style="background:none;border:none;color:#fff;font-size:1.5rem;cursor:pointer" onclick="closeLogsMo()">✕</button>
    </div>
    <div class="lb" id="lb">...</div>
  </div>
</div>

<!-- Person photos modal -->
<div class="pm" id="pm">
  <div class="pmh">
    <h3 id="pmTitle">Фото особи</h3>
    <span id="pmCount" style="font-size:.78rem;color:#64748b"></span>
    <div style="flex:1"></div>
    <label class="btn bp bs" for="pmFile" style="margin-right:8px;cursor:pointer">+ Фото</label>
    <input id="pmFile" type="file" accept="image/*" multiple style="display:none" onchange="enrollFromFiles()">
    <button style="background:none;border:none;color:#fff;font-size:1.5rem;cursor:pointer" onclick="closePm()">✕</button>
  </div>
  <div class="phg" id="phg"></div>
</div>

<!-- Photo preview modal -->
<div class="pp" id="pp" onclick="closePp()">
  <img id="ppImg" src="" onclick="event.stopPropagation()">
  <button class="ppdel" onclick="event.stopPropagation();deletePpPhoto()">✕</button>
  <button class="ppclose" onclick="event.stopPropagation();closePp()">✕</button>
</div>

<!-- Enroll snapshot dialog -->
<div class="mo" id="esDlg" onclick="closeEsDlg()" style="z-index:80">
  <div class="mdlg" style="min-width:280px" onclick="event.stopPropagation()">
    <h3>📥 Додати знімок до бази</h3>
    <div style="margin-bottom:10px">
      <label class="lbl">Особа</label>
      <select id="esSelPerson" onchange="esToggleNew()"></select>
    </div>
    <div id="esNewWrap" style="display:none;margin-bottom:10px">
      <label class="lbl">Нове ім'я (латиницею)</label>
      <input type="text" id="esNewName" placeholder="ім'я">
    </div>
    <div style="display:flex;gap:8px;justify-content:flex-end;margin-top:14px">
      <button class="btn bs" onclick="closeEsDlg()" style="background:#334155;color:#e2e8f0">Скасувати</button>
      <button class="btn bp bs" onclick="confirmEnrollSnap()">Додати</button>
    </div>
  </div>
</div>

<!-- Settings modal -->
<div class="mo" id="settingsMo" onclick="closeSettings()" style="z-index:55;align-items:flex-start;padding:20px">
  <div class="mdlg" style="width:min(640px,96vw)" onclick="event.stopPropagation()">
    <div style="display:flex;align-items:center;margin-bottom:14px">
      <h3 style="flex:1">⚙️ Налаштування</h3>
      <button style="background:none;border:none;color:#fff;font-size:1.5rem;cursor:pointer" onclick="closeSettings()">✕</button>
    </div>
    <div id="cfgBody">Завантаження...</div>
    <div style="display:flex;gap:8px;justify-content:flex-end;margin-top:16px">
      <button class="btn bs" onclick="closeSettings()" style="background:#334155;color:#e2e8f0">Скасувати</button>
      <button class="btn bp bs" onclick="saveSettings()">💾 Зберегти та перезапустити</button>
    </div>
  </div>
</div>

<script>
let autoT = null;
let persons = {};
let currentCamera = '';

// ── Groups (localStorage) ───────────────────────────────────────────────────
const GROUPS_KEY = 'fr2_groups';
const DEFAULT_GROUPS = [
  {id:'family', name:'Сімя', members:[]},
  {id:'neighbors', name:'Сусіди', members:[]},
  {id:'strangers', name:'Бродяги', members:[]},
];

function loadGroups(){
  try { return JSON.parse(localStorage.getItem(GROUPS_KEY)) || DEFAULT_GROUPS; }
  catch(e){ return DEFAULT_GROUPS; }
}
function saveGroups(g){ localStorage.setItem(GROUPS_KEY, JSON.stringify(g)); }
function groupOf(name){
  return loadGroups().find(g=>g.members.includes(name));
}
function assignGroup(name, gid){
  const gs = loadGroups();
  gs.forEach(g=>{ g.members = g.members.filter(m=>m!==name); });
  const tgt = gs.find(g=>g.id===gid);
  if(tgt) tgt.members.push(name);
  saveGroups(gs);
}
function removeFromGroups(name){
  const gs = loadGroups();
  gs.forEach(g=>{ g.members = g.members.filter(m=>m!==name); });
  saveGroups(gs);
}

// ── API ─────────────────────────────────────────────────────────────────────
async function api(url, opts){
  const r = await fetch(url, opts);
  return r.json();
}

// ── Camera ──────────────────────────────────────────────────────────────────
async function loadCameras(){
  try {
    const d = await api('/cameras');
    const sel = document.getElementById('camSel');
    sel.innerHTML = d.cameras.map(c=>`<option value="${c}">${c}</option>`).join('');
    if(!currentCamera && d.cameras.length) currentCamera = d.cameras[0];
    sel.value = currentCamera;
  } catch(e){ console.error('loadCameras',e); }
}
function onCamChange(){
  currentCamera = document.getElementById('camSel').value;
  loadPersons(); loadSnaps(); loadUnknown();
}

// ── Persons + columns ────────────────────────────────────────────────────────
async function loadPersons(){
  if(!currentCamera) return;
  try {
    const d = await api('/persons?camera='+encodeURIComponent(currentCamera));
    persons = d.persons || {};
    renderColumns();
    rebuildSnapFilter();
  } catch(e){ console.error('loadPersons',e); }
}

function renderColumns(){
  const wrap = document.getElementById('colsWrap');
  const gs = loadGroups();
  const assigned = new Set(gs.flatMap(g=>g.members));
  const unassigned = Object.keys(persons).filter(n=>!assigned.has(n)).sort();

  let html = gs.map(g=>`
    <div class="col-box">
      <div class="col-hdr">
        <span class="col-name" id="cname_${g.id}">${g.name}</span>
        <button title="Перейменувати" onclick="startRename('${g.id}')">✏️</button>
        <button title="Видалити стовпець" onclick="deleteCol('${g.id}')">✕</button>
      </div>
      <div class="col-persons" id="col_${g.id}">
        ${g.members.filter(n=>persons[n]!==undefined).map(n=>personCard(n,g.id)).join('')}
        ${g.members.filter(n=>persons[n]!==undefined).length===0?'<div class="empty" style="padding:8px 4px">порожньо</div>':''}
      </div>
    </div>`).join('');

  html += `<div>
    <div class="col-add-btn" onclick="addCol()">+ Новий стовпець</div>
    ${unassigned.length?`<div class="col-box" style="margin-top:8px;min-width:160px">
      <div class="col-hdr"><span class="col-name" style="color:#64748b">Не призначені</span></div>
      <div class="col-persons">${unassigned.map(n=>personCard(n,null)).join('')}</div>
    </div>`:''}
  </div>`;

  wrap.innerHTML = html;
}

function personCard(name, gid){
  const cnt = persons[name] || 0;
  // assigned: show remove-from-group button
  if(gid){
    return `<div class="pcard" onclick="openPm('${name}')">
      <span class="pav">👤</span>
      <div class="pinfo">
        <div class="pname">${name}</div>
        <div class="pcnt">${cnt} емб.</div>
      </div>
      <button class="pdel" title="Прибрати з групи" onclick="event.stopPropagation();removeFromGroups('${name}');renderColumns()">✕</button>
    </div>`;
  }
  // unassigned: show group assign buttons
  const gs = loadGroups();
  const btns = gs.map(g=>`<button style="padding:2px 6px;border-radius:5px;border:1px solid #334155;background:#1e293b;color:#94a3b8;font-size:.68rem;cursor:pointer;white-space:nowrap"
    onclick="event.stopPropagation();assignGroup('${name}','${g.id}');renderColumns()">${g.name}</button>`).join('');
  return `<div class="pcard" onclick="openPm('${name}')">
    <span class="pav">👤</span>
    <div class="pinfo">
      <div class="pname">${name}</div>
      <div class="pcnt">${cnt} емб.</div>
      <div style="display:flex;gap:3px;flex-wrap:wrap;margin-top:4px">${btns}</div>
    </div>
    <button class="pdel" title="Видалити з бази" onclick="event.stopPropagation();delPerson('${name}')">🗑</button>
  </div>`;
}

function startRename(gid){
  const el = document.getElementById('cname_'+gid);
  el.contentEditable = 'true';
  el.focus();
  const range = document.createRange();
  range.selectNodeContents(el);
  window.getSelection().removeAllRanges();
  window.getSelection().addRange(range);
  el.onblur = ()=>{
    const gs = loadGroups();
    const g = gs.find(x=>x.id===gid);
    if(g){ g.name = el.textContent.trim() || g.name; saveGroups(gs); }
    el.contentEditable='false';
    renderColumns();
  };
  el.onkeydown = e=>{ if(e.key==='Enter'){ e.preventDefault(); el.blur(); } };
}

function deleteCol(gid){
  if(!confirm('Видалити стовпець? Особи стануть "не призначеними"')) return;
  const gs = loadGroups().filter(g=>g.id!==gid);
  saveGroups(gs);
  renderColumns();
}

function addCol(){
  const name = prompt('Назва нового стовпця:','');
  if(!name) return;
  const gs = loadGroups();
  gs.push({id:'col_'+Date.now(), name:name.trim(), members:[]});
  saveGroups(gs);
  renderColumns();
}

function rebuildSnapFilter(){
  const sf = document.getElementById('sf');
  const cur = sf.value;
  sf.innerHTML = '<option value="">Всі особи</option>';
  for(const n of Object.keys(persons).sort()){
    sf.innerHTML += `<option value="${n}" ${n===cur?'selected':''}>${n}</option>`;
  }
}

// ── Persons CRUD ─────────────────────────────────────────────────────────────
async function delPerson(name){
  if(!confirm(`Видалити ${name} з бази ${currentCamera}?`)) return;
  await api('/persons/'+name+'?camera='+encodeURIComponent(currentCamera),{method:'DELETE'});
  removeFromGroups(name);
  loadPersons();
}
async function reloadDB(){
  await api('/reload?camera='+encodeURIComponent(currentCamera),{method:'POST'});
  setTimeout(loadPersons,300);
}

// ── Snapshots ────────────────────────────────────────────────────────────────
async function loadSnaps(){
  const f = document.getElementById('sf').value;
  try {
    const d = await api('/snapshots?person='+encodeURIComponent(f)+'&camera='+encodeURIComponent(currentCamera));
    const sg = document.getElementById('sg');
    if(!d.snapshots||!d.snapshots.length){ sg.innerHTML='<div class="empty">Немає знімків</div>'; return; }
    sg.innerHTML = d.snapshots.map(s=>`
      <div class="sc">
        <img src="/snapshots/img?path=${encodeURIComponent(s.path)}" loading="lazy" onerror="this.style.display='none'" onclick="openMo('/snapshots/img?path=${encodeURIComponent(s.path)}')">
        <div class="si">
          <div class="sn">${s.name}</div>
          <div class="sk">${s.camera} · ${s.time}</div>
          <button class="btn bp bs" style="margin-top:5px;width:100%;font-size:.72rem;padding:4px 8px" onclick="openEnrollSnapDlg('${s.name}','${s.path}',this)">📥 В базу</button>
        </div>
      </div>`).join('');
  } catch(e){ console.error('loadSnaps',e); }
}

function openMo(src){ document.getElementById('mi').src=src; document.getElementById('mo').classList.add('on'); }
function closeMo(){ document.getElementById('mo').classList.remove('on'); }

// ── Person photos modal ──────────────────────────────────────────────────────
let pmPerson = '';
async function openPm(name){
  pmPerson = name;
  document.getElementById('pmTitle').textContent = '👤 ' + name + ' · ' + currentCamera;
  document.getElementById('pm').classList.add('on');
  await loadPmPhotos();
}
function closePm(){ document.getElementById('pm').classList.remove('on'); pmPerson=''; }

async function loadPmPhotos(){
  const g = document.getElementById('phg');
  g.innerHTML = '<div class="empty" style="padding:16px">Завантаження...</div>';
  const d = await api(`/persons/${pmPerson}/photos/list?camera=${encodeURIComponent(currentCamera)}`);
  if(!d.files||!d.files.length){ g.innerHTML='<div class="empty" style="padding:16px">Фото відсутні</div>'; return; }
  document.getElementById('pmCount').textContent = d.files.length + ' фото';
  g.innerHTML = d.files.map(f=>`
    <div class="ph">
      <img src="/persons/${pmPerson}/photos/${encodeURIComponent(f)}/img?camera=${encodeURIComponent(currentCamera)}"
           onclick="openPp('/persons/${pmPerson}/photos/${encodeURIComponent(f)}/img?camera=${encodeURIComponent(currentCamera)}','${f}')"
           onerror="this.style.display='none'">
      <div class="pname">${f}</div>
    </div>`).join('');
}

async function enrollFromFiles(){
  const files = document.getElementById('pmFile').files;
  if(!files.length) return;
  let ok=0, fail=0;
  for(const file of files){
    const fd = new FormData();
    fd.append('image', file);
    try {
      const r = await fetch(`/persons/${encodeURIComponent(pmPerson)}/enroll?camera=${encodeURIComponent(currentCamera)}`,{method:'POST',body:fd});
      if(r.ok) ok++; else fail++;
    } catch(e){ fail++; }
  }
  document.getElementById('pmFile').value='';
  if(ok) showToast(`✓ Додано ${ok} фото`);
  else showToast('✗ Обличчя не знайдено', true);
  await loadPmPhotos();
  loadPersons();
}

let ppFile = '';
function openPp(src, fname){ ppFile=fname; document.getElementById('ppImg').src=src; document.getElementById('pp').classList.add('on'); }
function closePp(){ document.getElementById('pp').classList.remove('on'); ppFile=''; }
async function deletePpPhoto(){
  const fname=ppFile;
  if(!fname||!confirm(`Видалити ${fname}?`)) return;
  closePp();
  await api(`/persons/${pmPerson}/photos/${encodeURIComponent(fname)}?camera=${encodeURIComponent(currentCamera)}`,{method:'DELETE'});
  await loadPmPhotos(); loadPersons();
}

// ── Enroll snapshot dialog ───────────────────────────────────────────────────
let _esDlgPath='', _esDlgBtn=null;

function openEnrollSnapDlg(suggestedName, path, btn){
  _esDlgPath=path; _esDlgBtn=btn;
  const sel=document.getElementById('esSelPerson');
  const pnames=Object.keys(persons).sort();
  sel.innerHTML=pnames.map(p=>`<option value="${p}" ${p===suggestedName?'selected':''}>${p}</option>`).join('');
  sel.innerHTML+='<option value="__new__">+ Нова особа...</option>';
  if(!pnames.includes(suggestedName)) sel.value='__new__';
  esToggleNew();
  document.getElementById('esDlg').classList.add('on');
}
function closeEsDlg(){ document.getElementById('esDlg').classList.remove('on'); _esDlgPath=''; _esDlgBtn=null; }
function esToggleNew(){
  const v=document.getElementById('esSelPerson').value;
  document.getElementById('esNewWrap').style.display=v==='__new__'?'block':'none';
  if(v==='__new__') document.getElementById('esNewName').focus();
}
async function confirmEnrollSnap(){
  const sel=document.getElementById('esSelPerson').value;
  const name=sel==='__new__'?document.getElementById('esNewName').value.trim():sel;
  if(!name){ showToast('Введіть імʼя',true); return; }
  closeEsDlg();
  if(_esDlgBtn){ _esDlgBtn.disabled=true; _esDlgBtn.textContent='...'; }
  try {
    const r=await fetch(`/persons/${encodeURIComponent(name)}/enroll_snapshot?path=${encodeURIComponent(_esDlgPath)}&camera=${encodeURIComponent(currentCamera)}`,{method:'POST'});
    const d=await r.json();
    if(r.ok){ if(_esDlgBtn){_esDlgBtn.textContent='✓';_esDlgBtn.style.background='#22c55e';_esDlgBtn.style.color='#000';} showToast(`✓ ${name}: додано`); loadPersons(); }
    else { if(_esDlgBtn){_esDlgBtn.disabled=false;_esDlgBtn.textContent='📥';} showToast('✗ '+(d.error||'Помилка'),true); }
  } catch(e){ if(_esDlgBtn){_esDlgBtn.disabled=false;_esDlgBtn.textContent='📥';} showToast('✗ '+e.message,true); }
}

// ── Unknown ──────────────────────────────────────────────────────────────────
async function loadUnknown(){
  try {
    const d = await api('/unknown');
    const ug = document.getElementById('ug');
    const files = d.files || [];
    document.getElementById('ucnt').textContent = files.length ? files.length+' фото' : '';
    if(!files.length){ ug.innerHTML='<div class="empty">Немає невідомих облич</div>'; return; }
    const pnames = Object.keys(persons).sort();
    ug.innerHTML = files.map((f,i)=>{
      const opts = pnames.map(p=>`<option value="${p}">${p}</option>`).join('');
      return `<div class="uc">
        <img src="/unknown/${encodeURIComponent(f)}/img" onclick="openMo(this.src)" onerror="this.parentElement.remove()">
        <div class="ui">
          <div class="uk">${f}</div>
          <div class="ua">
            <select id="us${i}" onchange="toggleNewUn(${i})">
              ${opts}
              <option value="__new__">+ Нова особа...</option>
            </select>
            <button class="ub" onclick="assignUnknown('${f}',${i})">→</button>
            <button class="ud" onclick="deleteUnknown('${f}')">✕</button>
          </div>
          <input type="text" id="un${i}" placeholder="ім'я латиницею"
            style="display:${pnames.length===0?'block':'none'};margin-top:5px;width:100%;padding:4px 7px;border-radius:6px;border:1px solid #475569;background:#1e293b;color:#e2e8f0;font-size:.77rem">
        </div>
      </div>`;
    }).join('');
  } catch(e){ console.error('loadUnknown',e); }
}

function toggleNewUn(i){
  const v=document.getElementById('us'+i).value;
  document.getElementById('un'+i).style.display=v==='__new__'?'block':'none';
}

async function assignUnknown(fname, i){
  const sel=document.getElementById('us'+i);
  if(!sel) return;
  let person=sel.value;
  if(person==='__new__'){
    person=(document.getElementById('un'+i).value||'').trim();
    if(!person){ alert("Введіть ім'я"); return; }
  }
  try {
    const r=await fetch(`/unknown/${encodeURIComponent(fname)}/assign?person=${encodeURIComponent(person)}&camera=${encodeURIComponent(currentCamera)}`,{method:'POST'});
    const d=await r.json();
    if(r.ok){ showToast(d.warning?`⚠ ${person}: без ембедінгу`:`✓ ${person}: додано`); loadUnknown(); loadPersons(); }
    else showToast('✗ '+(d.error||'Помилка'),true);
  } catch(e){ showToast('✗ '+e.message,true); }
}

async function deleteUnknown(fname){
  if(!confirm(`Видалити ${fname}?`)) return;
  await api(`/unknown/${encodeURIComponent(fname)}`,{method:'DELETE'});
  loadUnknown();
}

// ── Snaps misc ───────────────────────────────────────────────────────────────
async function deleteAllSnaps(){
  if(!confirm('Видалити всі знімки?')) return;
  const r=await fetch('/snapshots',{method:'DELETE'});
  const d=await r.json();
  showToast(`🗑 Видалено ${d.removed} знімків`);
  loadSnaps();
}
function toggleAuto(){
  const btn=document.getElementById('ab');
  if(autoT){ clearInterval(autoT); autoT=null; btn.textContent='▶ Авто'; btn.style.background='#334155'; }
  else { autoT=setInterval(()=>{ loadSnaps(); },5000); btn.textContent='⏹ Стоп'; btn.style.background='#dc2626'; }
}

// ── Logs ──────────────────────────────────────────────────────────────────────
function openLogs(){ document.getElementById('logsMo').classList.add('on'); loadLogs(); }
function closeLogsMo(){ document.getElementById('logsMo').classList.remove('on'); }
async function loadLogs(){
  try {
    const d=await api('/logs');
    const lb=document.getElementById('lb');
    lb.textContent=d.lines||'(порожньо)';
    lb.scrollTop=lb.scrollHeight;
  } catch(e){}
}

// ── Health ───────────────────────────────────────────────────────────────────
async function checkHealth(){
  try {
    await api('/health');
    document.getElementById('dot').className='dot';
    document.getElementById('st').textContent='онлайн';
  } catch {
    document.getElementById('dot').className='dot off';
    document.getElementById('st').textContent='недоступний';
  }
}

// ── Toast ────────────────────────────────────────────────────────────────────
function showToast(msg,err){
  let t=document.getElementById('toast');
  if(!t){ t=document.createElement('div'); t.id='toast';
    t.style='position:fixed;bottom:20px;left:50%;transform:translateX(-50%);padding:10px 20px;border-radius:8px;font-size:.85rem;font-weight:600;z-index:999;transition:opacity .3s';
    document.body.appendChild(t); }
  t.textContent=msg; t.style.background=err?'#450a0a':'#14532d';
  t.style.color=err?'#fca5a5':'#86efac'; t.style.opacity='1';
  clearTimeout(t._t); t._t=setTimeout(()=>t.style.opacity='0',3000);
}

// ── Settings ─────────────────────────────────────────────────────────────────
let _cfgData = null;

async function openSettings(){
  document.getElementById('settingsMo').classList.add('on');
  document.getElementById('cfgBody').innerHTML = 'Завантаження...';
  try {
    const d = await api('/config');
    _cfgData = d;
    renderSettings(d);
  } catch(e){ document.getElementById('cfgBody').innerHTML = '✗ Помилка: '+e.message; }
}
function closeSettings(){ document.getElementById('settingsMo').classList.remove('on'); }

function cfgTab(camName){
  document.querySelectorAll('.cfg-tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.cfg-cam').forEach(t=>t.classList.remove('active'));
  document.getElementById('ctab_'+camName).classList.add('active');
  document.getElementById('ccam_'+camName).classList.add('active');
}

function renderSettings(d){
  const rec = d.recognition || {};
  const cameras = d.cameras || {};
  const camNames = Object.keys(cameras);

  const tabs = camNames.map((c,i)=>`<button class="cfg-tab${i===0?' active':''}" id="ctab_${c}" onclick="cfgTab('${c}')">${c}</button>`).join('');
  const camSections = camNames.map((c,i)=>camSection(c, cameras[c]||{}, i===0)).join('');

  document.getElementById('cfgBody').innerHTML = `
    <div class="cfg-section">
      <h4>Розпізнавання (глобально)</h4>
      <div class="cfg-grid">
        <div class="cfg-field"><label>similarity_threshold (впевнене розпізнавання)</label>
          <input type="number" id="r_sim" min="0.1" max="1" step="0.01" value="${rec.similarity_threshold||0.50}"></div>
        <div class="cfg-field"><label>unknown_threshold (мінімум для збереження)</label>
          <input type="number" id="r_unk" min="0.1" max="1" step="0.01" value="${rec.unknown_threshold||0.38}"></div>
        <div class="cfg-field"><label>cooldown_sec (пауза між сповіщеннями)</label>
          <input type="number" id="r_cd" min="1" max="3600" step="1" value="${rec.cooldown_sec||30}"></div>
        <div class="cfg-field"><label>det_score_min (мінімум детектора)</label>
          <input type="number" id="r_det" min="0.1" max="1" step="0.01" value="${rec.det_score_min||0.3}"></div>
        <div class="cfg-field"><label>det_size (розмір детекції)</label>
          <select id="r_ds">
            ${[160,320,640].map(v=>`<option value="${v}" ${rec.det_size==v?'selected':''}>${v}×${v}</option>`).join('')}
          </select></div>
      </div>
    </div>
    <div class="cfg-section">
      <h4>Камери</h4>
      <div class="cfg-tabs" id="cfgTabs">${tabs}
        <button class="cfg-tab" onclick="addCameraForm()" style="border-style:dashed;color:#6366f1">+ Додати</button>
      </div>
      <div id="cfgCams">${camSections}</div>
    </div>`;
}

function camSection(c, cc, active){
  const g = cc.gesture || {};
  return `<div class="cfg-cam${active?' active':''}" id="ccam_${c}">
    <div class="cfg-grid">
      <div class="cfg-field" style="grid-column:1/-1"><label>RTSP (основний потік)</label>
        <input type="text" id="c_${c}_rtsp" value="${cc.rtsp||''}"></div>
      <div class="cfg-field" style="grid-column:1/-1"><label>RTSP HD (знімки, необов'язково)</label>
        <input type="text" id="c_${c}_rtsp_hd" value="${cc.rtsp_hd||''}"></div>
      <div class="cfg-field"><label>fps_process (кадрів/сек)</label>
        <input type="number" id="c_${c}_fps" min="1" max="10" step="1" value="${cc.fps_process||2}"></div>
      <div class="cfg-field"><label>min_face_w (мін. ширина обличчя, px)</label>
        <input type="number" id="c_${c}_mfw" min="0" max="300" step="1" value="${cc.min_face_w||0}"></div>
      <div class="cfg-field"><label>max_pitch (макс. кут нахилу голови °)</label>
        <input type="number" id="c_${c}_mp" min="10" max="90" step="1" value="${cc.max_pitch||35}"></div>
      <div class="cfg-field"><label>rotation (поворот камери °)</label>
        <select id="c_${c}_rot">
          ${[0,90,180,270].map(v=>`<option value="${v}" ${cc.rotation==v?'selected':''}>${v}°</option>`).join('')}
        </select></div>
    </div>
    <div style="margin-top:12px">
      <label style="display:flex;align-items:center;gap:8px;font-size:.78rem;color:#94a3b8;cursor:pointer">
        <input type="checkbox" id="c_${c}_gen" ${g.enabled?'checked':''} onchange="toggleGesture('${c}')">
        Розпізнавання жестів (великий палець → trigger)
      </label>
      <div id="c_${c}_gbox" style="margin-top:8px;display:${g.enabled?'grid':'none'};grid-template-columns:1fr 1fr;gap:10px">
        <div class="cfg-field"><label>gesture fps_process</label>
          <input type="number" id="c_${c}_gfps" min="1" max="5" step="1" value="${g.fps_process||2}"></div>
        <div class="cfg-field"><label>gesture cooldown (сек)</label>
          <input type="number" id="c_${c}_gcd" min="1" max="60" step="1" value="${g.cooldown_sec||5}"></div>
      </div>
    </div>
    <div style="margin-top:14px;text-align:right">
      <button class="btn br bs" onclick="removeCamera('${c}')">🗑 Видалити камеру ${c}</button>
    </div>
  </div>`;
}

function addCameraForm(){
  const name = prompt('Назва камери (напр. cam_66):','');
  if(!name || !name.trim()) return;
  const c = name.trim();
  if(_cfgData.cameras[c]){ showToast('✗ Камера вже існує', true); return; }
  _cfgData.cameras[c] = {rtsp:'', fps_process:2, min_face_w:0, rotation:0};
  // add tab
  const tabs = document.getElementById('cfgTabs');
  const addBtn = tabs.querySelector('button:last-child');
  const tab = document.createElement('button');
  tab.className = 'cfg-tab';
  tab.id = 'ctab_'+c;
  tab.textContent = c;
  tab.onclick = ()=>cfgTab(c);
  tabs.insertBefore(tab, addBtn);
  // add section
  document.getElementById('cfgCams').insertAdjacentHTML('beforeend', camSection(c, _cfgData.cameras[c], false));
  cfgTab(c);
}

function removeCamera(c){
  if(!confirm(`Видалити камеру "${c}" з конфігу?\nПотоки зупиняться після перезапуску.`)) return;
  delete _cfgData.cameras[c];
  // remove tab and section
  const tab = document.getElementById('ctab_'+c);
  const sec = document.getElementById('ccam_'+c);
  if(tab) tab.remove();
  if(sec) sec.remove();
  // activate first remaining tab
  const first = document.querySelector('.cfg-tab:not([onclick*="addCameraForm"])');
  if(first) first.click();
}

function toggleGesture(cam){
  const on = document.getElementById('c_'+cam+'_gen').checked;
  document.getElementById('c_'+cam+'_gbox').style.display = on ? 'grid' : 'none';
}

function _collectCameras(){
  const cameras = {};
  for(const c of Object.keys(_cfgData.cameras)){
    const rtsp = document.getElementById('c_'+c+'_rtsp')?.value?.trim();
    if(!rtsp) continue; // skip if rtsp empty (new camera not filled)
    const rtsp_hd = document.getElementById('c_'+c+'_rtsp_hd')?.value?.trim();
    cameras[c] = {
      rtsp,
      ...(rtsp_hd ? {rtsp_hd} : {}),
      fps_process: parseInt(document.getElementById('c_'+c+'_fps').value),
      min_face_w:  parseInt(document.getElementById('c_'+c+'_mfw').value),
      max_pitch:   parseInt(document.getElementById('c_'+c+'_mp').value),
      rotation:    parseInt(document.getElementById('c_'+c+'_rot').value),
    };
    // keep original fields not in UI (e.g. mqtt-specific)
    const orig = _cfgData.cameras[c] || {};
    Object.keys(orig).forEach(k=>{ if(!(k in cameras[c]) && !['gesture'].includes(k)) cameras[c][k]=orig[k]; });
    const genOn = document.getElementById('c_'+c+'_gen').checked;
    if(genOn){
      cameras[c].gesture = {
        enabled: true,
        fps_process: parseInt(document.getElementById('c_'+c+'_gfps').value),
        cooldown_sec: parseInt(document.getElementById('c_'+c+'_gcd').value),
      };
    }
  }
  return cameras;
}

async function saveSettings(){
  if(!_cfgData) return;
  const newCfg = JSON.parse(JSON.stringify(_cfgData));
  newCfg.recognition.similarity_threshold = parseFloat(document.getElementById('r_sim').value);
  newCfg.recognition.unknown_threshold    = parseFloat(document.getElementById('r_unk').value);
  newCfg.recognition.cooldown_sec         = parseInt(document.getElementById('r_cd').value);
  newCfg.recognition.det_score_min        = parseFloat(document.getElementById('r_det').value);
  newCfg.recognition.det_size             = parseInt(document.getElementById('r_ds').value);
  newCfg.cameras = _collectCameras();
  if(!Object.keys(newCfg.cameras).length){ showToast('✗ Потрібна хоча б одна камера', true); return; }
  try {
    const r = await fetch('/config', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(newCfg)});
    const d = await r.json();
    if(r.ok){ closeSettings(); showToast('✓ Збережено, перезапуск...'); }
    else showToast('✗ '+(d.error||'Помилка'), true);
  } catch(e){ showToast('✗ '+e.message, true); }
}

// ── Init ─────────────────────────────────────────────────────────────────────
checkHealth();
loadCameras().then(()=>{ loadPersons(); loadSnaps(); loadUnknown(); });
setInterval(checkHealth,15000);
</script>
</body>
</html>"""


def create_app(face_dbs, snapshot_dir=SNAPSHOT_DIR, cameras=None):
    """face_dbs: dict {camera_name: FaceDB}"""
    app = Flask(__name__)
    unknown_dir = os.path.join(os.path.dirname(snapshot_dir), "unknown")
    _rebuild_lock = threading.Lock()
    _cam_names = list(face_dbs.keys())

    def _db(request):
        cam = request.args.get("camera", _cam_names[0] if _cam_names else "")
        return face_dbs.get(cam) or face_dbs[_cam_names[0]]

    def _rebuild_bg(face_db, name):
        if _rebuild_lock.acquire(blocking=False):
            try:
                face_db.rebuild_person(name)
            finally:
                _rebuild_lock.release()

    @app.route("/")
    def index():
        return Response(HTML, mimetype="text/html; charset=utf-8")

    @app.route("/cameras", methods=["GET"])
    def list_cameras():
        return jsonify({"cameras": _cam_names})

    @app.route("/persons", methods=["GET"])
    def list_persons():
        return jsonify({"persons": _db(request).list_persons()})

    @app.route("/persons/<name>", methods=["DELETE"])
    def delete_person(name):
        ok = _db(request).delete_person(name)
        if ok:
            return jsonify({"status": "deleted", "name": name})
        return jsonify({"error": "not found"}), 404

    @app.route("/persons/<name>/enroll", methods=["POST"])
    def enroll(name):
        if "image" not in request.files:
            return jsonify({"error": "missing 'image' field"}), 400
        data = request.files["image"].read()
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "cannot decode image"}), 400
        count = _db(request).enroll(name, img)
        if count == 0:
            return jsonify({"error": "no face detected"}), 422
        return jsonify({"status": "enrolled", "name": name, "total": count})

    @app.route("/persons/<name>/photos", methods=["GET"])
    def list_photos(name):
        return jsonify({"name": name, "photos": _db(request).photo_count(name)})

    @app.route("/persons/<name>/photos/list", methods=["GET"])
    def list_photo_files(name):
        person_dir = os.path.join(_db(request).faces_dir, name)
        if not os.path.isdir(person_dir):
            return jsonify({"error": "not found"}), 404
        files = sorted(f for f in os.listdir(person_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        return jsonify({"name": name, "files": files})

    @app.route("/persons/<name>/photos/<fname>", methods=["DELETE"])
    def delete_photo(name, fname):
        db = _db(request)
        person_dir = os.path.join(db.faces_dir, name)
        path = os.path.join(person_dir, fname)
        if not os.path.realpath(path).startswith(os.path.realpath(person_dir)):
            return jsonify({"error": "forbidden"}), 403
        if not os.path.isfile(path):
            return jsonify({"error": "not found"}), 404
        os.remove(path)
        threading.Thread(target=_rebuild_bg, args=(db, name,), daemon=True).start()
        remaining = len([f for f in os.listdir(person_dir)
                         if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        return jsonify({"status": "deleted", "remaining": remaining})

    @app.route("/persons/<name>/photos/<fname>/img", methods=["GET"])
    def photo_img(name, fname):
        person_dir = os.path.join(_db(request).faces_dir, name)
        path = os.path.join(person_dir, fname)
        if not os.path.realpath(path).startswith(os.path.realpath(person_dir)):
            return "", 403
        if not os.path.isfile(path):
            return "", 404
        return send_file(path, mimetype="image/jpeg")

    @app.route("/unknown", methods=["GET"])
    def list_unknown():
        if not os.path.isdir(unknown_dir):
            return jsonify({"files": []})
        files = sorted(
            (f for f in os.listdir(unknown_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))),
            reverse=True,
        )
        return jsonify({"files": files})

    @app.route("/unknown/<fname>/img", methods=["GET"])
    def unknown_img(fname):
        path = os.path.join(unknown_dir, fname)
        if not os.path.realpath(path).startswith(os.path.realpath(unknown_dir)):
            return "", 403
        if not os.path.isfile(path):
            return "", 404
        return send_file(path, mimetype="image/jpeg")

    @app.route("/unknown/<fname>/assign", methods=["POST"])
    def assign_unknown(fname):
        person = request.args.get("person", "").strip()
        if not person:
            return jsonify({"error": "missing person"}), 400
        src = os.path.join(unknown_dir, fname)
        if not os.path.realpath(src).startswith(os.path.realpath(unknown_dir)):
            return jsonify({"error": "forbidden"}), 403
        if not os.path.isfile(src):
            return jsonify({"error": "not found"}), 404
        img = cv2.imread(src)
        if img is None:
            return jsonify({"error": "cannot decode image"}), 400
        db = _db(request)
        count = db.enroll(person, img)
        # move file to person folder regardless of enrollment result
        import shutil
        person_dir = os.path.join(db.faces_dir, person)
        os.makedirs(person_dir, exist_ok=True)
        dst = os.path.join(person_dir, os.path.basename(src))
        shutil.move(src, dst)
        if count == 0:
            return jsonify({"status": "moved", "person": person, "warning": "no face detected, photo saved without embedding"})
        return jsonify({"status": "assigned", "person": person, "total": count})

    @app.route("/unknown/<fname>", methods=["DELETE"])
    def delete_unknown(fname):
        path = os.path.join(unknown_dir, fname)
        if not os.path.realpath(path).startswith(os.path.realpath(unknown_dir)):
            return jsonify({"error": "forbidden"}), 403
        if not os.path.isfile(path):
            return jsonify({"error": "not found"}), 404
        os.remove(path)
        return jsonify({"status": "deleted"})

    @app.route("/persons/<name>/enroll_snapshot", methods=["POST"])
    def enroll_snapshot(name):
        path = request.args.get("path", "")
        if not os.path.realpath(path).startswith(os.path.realpath(snapshot_dir)):
            return jsonify({"error": "invalid path"}), 400
        if not os.path.isfile(path):
            return jsonify({"error": "file not found"}), 404
        img = cv2.imread(path)
        if img is None:
            return jsonify({"error": "cannot decode image"}), 400
        count = _db(request).enroll(name, img)
        if count == 0:
            return jsonify({"error": "no face detected in snapshot"}), 422
        return jsonify({"status": "enrolled", "name": name, "total": count})

    @app.route("/reload", methods=["POST"])
    def reload_db():
        _db(request).reload()
        return jsonify({"status": "reloaded"})

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    @app.route("/snapshots", methods=["GET"])
    def list_snapshots():
        person_filter = request.args.get("person", "").lower()
        items = []
        if os.path.isdir(snapshot_dir):
            for day in sorted(os.listdir(snapshot_dir), reverse=True):
                day_dir = os.path.join(snapshot_dir, day)
                if not os.path.isdir(day_dir):
                    continue
                for fname in sorted(os.listdir(day_dir), reverse=True):
                    if not fname.lower().endswith(".jpg"):
                        continue
                    camera, name, ts = _parse_snap_filename(fname)
                    if person_filter and name.lower() != person_filter:
                        continue
                    t = f"{ts[:2]}:{ts[2:4]}:{ts[4:6]}" if len(ts) >= 6 else ts
                    items.append({
                        "path": os.path.join(day_dir, fname),
                        "name": name,
                        "camera": camera,
                        "time": f"{day} {t}",
                    })
                    if len(items) >= 60:
                        break
                if len(items) >= 60:
                    break
        return jsonify({"snapshots": items})

    @app.route("/snapshots", methods=["DELETE"])
    def delete_all_snapshots():
        removed = 0
        if os.path.isdir(snapshot_dir):
            for day in os.listdir(snapshot_dir):
                day_dir = os.path.join(snapshot_dir, day)
                if not os.path.isdir(day_dir):
                    continue
                for fname in os.listdir(day_dir):
                    fpath = os.path.join(day_dir, fname)
                    if os.path.isfile(fpath):
                        os.remove(fpath)
                        removed += 1
                try:
                    os.rmdir(day_dir)
                except OSError:
                    pass
        return jsonify({"status": "ok", "removed": removed})

    @app.route("/snapshots/img")
    def snapshot_img():
        path = request.args.get("path", "")
        if not path.startswith(snapshot_dir) or not os.path.isfile(path):
            return "", 404
        return send_file(path, mimetype="image/jpeg")

    @app.route("/logs")
    def get_logs():
        try:
            result = subprocess.run(
                ["journalctl", "-u", "facerecog2", "-n", "80",
                 "--no-pager", "--output=short"],
                capture_output=True, text=True, timeout=5
            )
            return jsonify({"lines": result.stdout})
        except Exception as e:
            return jsonify({"lines": str(e)})

    _config_path = os.environ.get("FR2_CONFIG", "/opt/facerecog2/app/config.yml")

    @app.route("/config", methods=["GET"])
    def get_config():
        try:
            with open(_config_path) as f:
                cfg = yaml.safe_load(f)
            return jsonify(cfg)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/config", methods=["POST"])
    def save_config():
        try:
            cfg = request.get_json(force=True)
            if not cfg:
                return jsonify({"error": "empty body"}), 400
            with open(_config_path, "w") as f:
                yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            # restart service after short delay so response is sent first
            def _restart():
                import time; time.sleep(1)
                subprocess.run(["systemctl", "restart", "facerecog2"], check=False)
            threading.Thread(target=_restart, daemon=True).start()
            return jsonify({"status": "saved"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/trigger/<camera_name>", methods=["POST"])
    def trigger(camera_name):
        """Trigger immediate recognition on a camera (called by Home Assistant on motion)."""
        if cameras is None or camera_name not in cameras:
            return jsonify({"error": f"camera '{camera_name}' not found"}), 404
        cam = cameras[camera_name]
        threading.Thread(target=cam.trigger, daemon=True).start()
        return jsonify({"status": "triggered", "camera": camera_name}), 202

    @app.route("/latest/<camera_name>")
    def latest_snapshot(camera_name):
        """Return the most recent snapshot JPEG for a camera (for HA picture card)."""
        day_dirs = sorted([
            d for d in os.listdir(snapshot_dir)
            if os.path.isdir(os.path.join(snapshot_dir, d))
        ], reverse=True)
        for day in day_dirs:
            day_path = os.path.join(snapshot_dir, day)
            files = sorted([
                f for f in os.listdir(day_path)
                if f.startswith(camera_name) and f.endswith(".jpg")
            ], reverse=True)
            if files:
                path = os.path.join(day_path, files[0])
                return send_file(path, mimetype="image/jpeg")
        return Response(status=404)

    return app


def run_api(face_dbs, host, port, snapshot_dir=SNAPSHOT_DIR, cameras=None):
    flask_app = create_app(face_dbs, snapshot_dir, cameras)
    t = threading.Thread(
        target=lambda: flask_app.run(host=host, port=port, debug=False, use_reloader=False),
        daemon=True,
        name="api",
    )
    t.start()
    log.info("Web UI on http://%s:%s", host, port)
