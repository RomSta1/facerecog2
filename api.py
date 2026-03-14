import base64
import logging
import os
import subprocess
import threading

import cv2
import numpy as np
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
header{background:#1e293b;padding:14px 24px;display:flex;align-items:center;gap:12px;border-bottom:1px solid #334155;position:sticky;top:0;z-index:10}
header h1{font-size:1.2rem;font-weight:700}
.dot{width:9px;height:9px;border-radius:50%;background:#22c55e}
.dot.off{background:#ef4444}
main{max-width:1100px;margin:0 auto;padding:20px;display:grid;gap:20px}
.card{background:#1e293b;border-radius:12px;padding:20px;border:1px solid #334155}
.card h2{font-size:.8rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:.06em;margin-bottom:14px}
/* persons */
.pg{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:10px}
.pc{background:#0f172a;border-radius:8px;padding:12px;text-align:center;border:1px solid #334155;position:relative}
.pc .dn{font-weight:600;font-size:.95rem;word-break:break-all}
.pc .dc{font-size:.75rem;color:#64748b;margin-top:3px}
.pc .dx{position:absolute;top:6px;right:6px;background:none;border:none;color:#ef4444;cursor:pointer;font-size:.85rem;opacity:.5;line-height:1}
.pc .dx:hover{opacity:1}
/* enroll */
.ef{display:grid;grid-template-columns:1fr 1fr auto;gap:10px;align-items:end}
@media(max-width:600px){.ef{grid-template-columns:1fr}}
label.lbl{font-size:.78rem;color:#94a3b8;display:block;margin-bottom:4px}
input[type=text],select{width:100%;background:#0f172a;border:1px solid #334155;color:#e2e8f0;padding:9px 12px;border-radius:8px;font-size:.9rem;outline:none}
input[type=text]:focus,select:focus{border-color:#6366f1}
.frow{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.fl{display:inline-block;padding:9px 14px;border-radius:8px;border:1px dashed #475569;cursor:pointer;font-size:.85rem;color:#94a3b8;white-space:nowrap}
.fl:hover{border-color:#6366f1;color:#a5b4fc}
#fi{display:none}
#fn{font-size:.8rem;color:#64748b}
.btn{padding:9px 18px;border-radius:8px;border:none;font-size:.88rem;font-weight:600;cursor:pointer;white-space:nowrap}
.bp{background:#6366f1;color:#fff}.bp:hover{background:#4f46e5}
.bg{background:#22c55e;color:#000}.bg:hover{background:#16a34a}
.bs{padding:6px 12px;font-size:.8rem}
.msg{padding:9px 13px;border-radius:8px;font-size:.83rem;margin-top:10px}
.ok{background:#14532d;color:#86efac}
.er{background:#450a0a;color:#fca5a5}
/* snaps */
.sg{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:10px;max-height:460px;overflow-y:auto}
.sc{background:#0f172a;border-radius:8px;overflow:hidden;border:1px solid #334155;cursor:pointer}
.sc:hover{border-color:#6366f1}
.sc img{width:100%;height:115px;object-fit:cover;display:block}
.si{padding:7px 8px;font-size:.73rem}
.sn{font-weight:600;color:#f1f5f9;font-size:.82rem}
.sk{color:#64748b;margin-top:2px}
/* logs */
.lb{background:#020617;border-radius:8px;padding:12px;font-family:monospace;font-size:.72rem;color:#94a3b8;height:200px;overflow-y:auto;white-space:pre-wrap;border:1px solid #1e293b;line-height:1.5}
/* snap modal */
.mo{display:none;position:fixed;inset:0;background:rgba(0,0,0,.85);z-index:50;align-items:center;justify-content:center}
.mo.on{display:flex}
.mo img{max-width:92vw;max-height:92vh;border-radius:6px}
.mc{position:fixed;top:12px;right:16px;background:none;border:none;color:#fff;font-size:2rem;cursor:pointer;line-height:1}
/* unknown faces */
.ug{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px;max-height:380px;overflow-y:auto}
.uc{background:#0f172a;border-radius:8px;overflow:hidden;border:1px solid #334155}
.uc img{width:100%;height:110px;object-fit:cover;display:block;cursor:pointer}
.ui{padding:7px 8px}
.uk{color:#64748b;font-size:.7rem;margin-bottom:6px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.ua{display:flex;gap:4px;align-items:center}
.ua select{flex:1;padding:5px 6px;font-size:.75rem;background:#1e293b;border:1px solid #475569;color:#e2e8f0;border-radius:6px}
.ua button{padding:5px 8px;border-radius:6px;border:none;font-size:.75rem;cursor:pointer;font-weight:600;white-space:nowrap}
.ub{background:#6366f1;color:#fff}.ub:hover{background:#4f46e5}
.ud{background:#334155;color:#ef4444}.ud:hover{background:#450a0a}
/* person photos modal */
.pm{display:none;position:fixed;inset:0;background:rgba(0,0,0,.92);z-index:60;flex-direction:column}
.pm.on{display:flex}
.pmh{background:#1e293b;padding:14px 20px;display:flex;align-items:center;gap:12px;border-bottom:1px solid #334155}
.pmh h3{font-size:1rem;font-weight:700;flex:1}
.phg{display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:10px;padding:16px;overflow-y:auto;flex:1}
.ph{position:relative;background:#0f172a;border-radius:8px;overflow:hidden;border:1px solid #334155}
.ph img{width:100%;height:130px;object-fit:cover;display:block;cursor:pointer}
.ph .pname{padding:4px 6px;font-size:.72rem;color:#94a3b8;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
/* photo preview modal */
.pp{display:none;position:fixed;inset:0;background:rgba(0,0,0,.92);z-index:70;align-items:center;justify-content:center}
.pp.on{display:flex}
.pp img{max-width:92vw;max-height:88vh;border-radius:6px;object-fit:contain}
.pp .ppdel{position:fixed;top:14px;right:56px;background:rgba(239,68,68,.9);border:none;color:#fff;border-radius:50%;width:34px;height:34px;font-size:1.1rem;cursor:pointer;display:flex;align-items:center;justify-content:center}
.pp .ppdel:hover{background:#dc2626}
.pp .ppclose{position:fixed;top:12px;right:16px;background:none;border:none;color:#fff;font-size:2rem;cursor:pointer;line-height:1}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:12px}
.empty{color:#475569;font-size:.85rem;padding:8px 0}
</style>
</head>
<body>
<header>
  <div class="dot" id="dot"></div>
  <h1>FaceRecog2</h1>
  <span id="st" style="font-size:.8rem;color:#64748b">...</span>
  <div style="flex:1"></div>
  <button class="btn bp bs" onclick="reloadDB()">↺ Перезавантажити базу</button>
</header>
<main>

<div class="card">
  <h2>Невідомі обличчя</h2>
  <div class="row">
    <button class="btn bp bs" onclick="loadUnknown()">Оновити</button>
    <span id="ucnt" style="font-size:.78rem;color:#64748b"></span>
  </div>
  <div class="ug" id="ug"><div class="empty">Немає невідомих облич</div></div>
</div>

<div class="card">
  <h2>Особи в базі</h2>
  <div class="pg" id="pg"><div class="empty">Завантаження...</div></div>
</div>

<div class="card">
  <h2>Додати фото особи</h2>
  <div class="ef">
    <div>
      <label class="lbl">Вибрати особу</label>
      <select id="selPerson" onchange="toggleNew()">
        <option value="">-- завантаження --</option>
        <option value="__new__">+ Нова особа...</option>
      </select>
    </div>
    <div id="newNameWrap" style="display:none">
      <label class="lbl">Ім'я нової особи</label>
      <input type="text" id="newName" placeholder="ім'я латиницею">
    </div>
    <div>
      <label class="lbl">Фото</label>
      <div class="frow">
        <label class="fl" for="fi">📁 Обрати</label>
        <input id="fi" type="file" accept="image/*" multiple onchange="onFiles()">
        <span id="fn"></span>
      </div>
    </div>
  </div>
  <div style="margin-top:12px">
    <button class="btn bp" onclick="doEnroll()">Додати фото</button>
  </div>
  <div id="em"></div>
</div>

<div class="card">
  <h2>Останні знімки</h2>
  <div class="row">
    <select id="sf" onchange="loadSnaps()" style="background:#0f172a;border:1px solid #334155;color:#e2e8f0;padding:7px 10px;border-radius:8px;font-size:.85rem">
      <option value="">Всі особи</option>
    </select>
    <button class="btn bp bs" onclick="loadSnaps()">Оновити</button>
    <button class="btn bs" id="ab" onclick="toggleAuto()" style="background:#334155;color:#e2e8f0">▶ Авто</button>
    <button class="btn bs" onclick="deleteAllSnaps()" style="background:#450a0a;color:#fca5a5;margin-left:auto">🗑 Видалити всі</button>
  </div>
  <div class="sg" id="sg"><div class="empty">Немає знімків</div></div>
</div>

<div class="card">
  <h2>Логи сервісу</h2>
  <div class="row">
    <button class="btn bp bs" onclick="loadLogs()">Оновити</button>
  </div>
  <div class="lb" id="lb">...</div>
</div>

</main>
<div class="mo" id="mo" onclick="closeMo()">
  <button class="mc" onclick="closeMo()">✕</button>
  <img id="mi" src="">
</div>

<!-- Person photos modal -->
<div class="pm" id="pm">
  <div class="pmh">
    <h3 id="pmTitle">Фото особи</h3>
    <span id="pmCount" style="font-size:.8rem;color:#64748b"></span>
    <div style="flex:1"></div>
    <button class="btn bp bs" style="margin-right:8px" onclick="openEnrollForPerson()">+ Додати фото</button>
    <button class="mc" style="position:static;font-size:1.5rem" onclick="closePm()">✕</button>
  </div>
  <div class="phg" id="phg"></div>
</div>

<!-- Photo preview modal -->
<div class="pp" id="pp" onclick="closePp()">
  <img id="ppImg" src="" onclick="event.stopPropagation()">
  <button class="ppdel" onclick="event.stopPropagation();deletePpPhoto()">✕</button>
  <button class="ppclose" onclick="event.stopPropagation();closePp()">✕</button>
</div>

<script>
let autoT = null;
let persons = {};

async function api(url, opts){
  const r = await fetch(url, opts);
  return r.json();
}

async function loadPersons(){
  try {
    const d = await api('/persons');
    persons = d.persons || {};
    const pg = document.getElementById('pg');
    const sel = document.getElementById('selPerson');
    const sf = document.getElementById('sf');
    const sfv = sf.value;

    pg.innerHTML = '';
    if(!Object.keys(persons).length){
      pg.innerHTML = '<div class="empty">База порожня</div>';
    } else {
      for(const [n,c] of Object.entries(persons).sort()){
        const d = document.createElement('div');
        d.className = 'pc';
        d.style.cursor = 'pointer';
        d.innerHTML = `<button class="dx" onclick="event.stopPropagation();delPerson('${n}')">✕</button>
          <div style="font-size:1.6rem;margin-bottom:6px">👤</div>
          <div class="dn">${n}</div><div class="dc">${c} емб.</div>
          <div style="font-size:.7rem;color:#475569;margin-top:4px">натисни → фото</div>`;
        d.onclick = () => openPm(n);
        pg.appendChild(d);
      }
    }

    // rebuild person select
    const cur = sel.value;
    sel.innerHTML = '';
    for(const n of Object.keys(persons).sort()){
      sel.innerHTML += `<option value="${n}">${n}</option>`;
    }
    sel.innerHTML += '<option value="__new__">+ Нова особа...</option>';
    if(cur && [...sel.options].some(o=>o.value===cur)) sel.value = cur;
    toggleNew();

    // rebuild snap filter
    sf.innerHTML = '<option value="">Всі особи</option>';
    for(const n of Object.keys(persons).sort()){
      sf.innerHTML += `<option value="${n}" ${n===sfv?'selected':''}>${n}</option>`;
    }
  } catch(e){ console.error('loadPersons',e); }
}

function toggleNew(){
  const v = document.getElementById('selPerson').value;
  document.getElementById('newNameWrap').style.display = v==='__new__' ? 'block' : 'none';
}

function onFiles(){
  const files = document.getElementById('fi').files;
  document.getElementById('fn').textContent = files.length ? `${files.length} файл(ів)` : '';
}

async function doEnroll(){
  const sel = document.getElementById('selPerson').value;
  const name = sel==='__new__'
    ? document.getElementById('newName').value.trim()
    : sel;
  const files = document.getElementById('fi').files;
  const msg = document.getElementById('em');
  msg.className=''; msg.textContent='';

  if(!name){ showMsg('er','Оберіть або введіть ім\'я'); return; }
  if(!files.length){ showMsg('er','Оберіть хоча б один файл'); return; }

  let ok=0, fail=0;
  for(const file of files){
    const fd = new FormData();
    fd.append('image', file);
    try {
      const r = await fetch(`/persons/${encodeURIComponent(name)}/enroll`,{method:'POST',body:fd});
      const d = await r.json();
      if(r.ok) ok++;
      else { fail++; console.warn(d); }
    } catch(e){ fail++; }
  }

  if(ok) showMsg('ok', `✓ Додано ${ok} фото${fail?' ('+fail+' не вдалось)':''}. Ембедінгів: ?`);
  else showMsg('er','✗ Жодне фото не прийнято — обличчя не виявлено');
  loadPersons();
  document.getElementById('fi').value='';
  document.getElementById('fn').textContent='';
}

function showMsg(cls,txt){
  const m=document.getElementById('em');
  m.className='msg '+cls; m.textContent=txt;
  setTimeout(()=>{m.className='';m.textContent='';},5000);
}

async function delPerson(name){
  if(!confirm(`Видалити ${name} з бази?`)) return;
  await api('/persons/'+name,{method:'DELETE'});
  loadPersons();
}

async function reloadDB(){
  await api('/reload',{method:'POST'});
  setTimeout(loadPersons,300);
}

async function loadSnaps(){
  const f = document.getElementById('sf').value;
  try {
    const d = await api('/snapshots?person='+encodeURIComponent(f));
    const sg = document.getElementById('sg');
    if(!d.snapshots||!d.snapshots.length){
      sg.innerHTML='<div class="empty">Немає знімків</div>'; return;
    }
    sg.innerHTML = d.snapshots.map(s=>`
      <div class="sc" onclick="openMo('/snapshots/img?path=${encodeURIComponent(s.path)}')">
        <img src="/snapshots/img?path=${encodeURIComponent(s.path)}" loading="lazy" onerror="this.style.display='none'">
        <div class="si"><div class="sn">${s.name}</div><div class="sk">${s.camera} · ${s.time}</div></div>
      </div>`).join('');
  } catch(e){ console.error('loadSnaps',e); }
}

function openMo(src){ document.getElementById('mi').src=src; document.getElementById('mo').classList.add('on'); }
function closeMo(){ document.getElementById('mo').classList.remove('on'); }

let pmPerson = '';
async function openPm(name){
  pmPerson = name;
  document.getElementById('pmTitle').textContent = '👤 ' + name;
  document.getElementById('pm').classList.add('on');
  await loadPmPhotos();
}
function closePm(){ document.getElementById('pm').classList.remove('on'); pmPerson=''; }

async function loadPmPhotos(){
  const g = document.getElementById('phg');
  g.innerHTML = '<div class="empty">Завантаження...</div>';
  const d = await api(`/persons/${pmPerson}/photos/list`);
  if(!d.files||!d.files.length){ g.innerHTML='<div class="empty">Фото відсутні</div>'; return; }
  document.getElementById('pmCount').textContent = d.files.length + ' фото';
  g.innerHTML = d.files.map(f=>`
    <div class="ph">
      <img src="/persons/${pmPerson}/photos/${encodeURIComponent(f)}/img"
           onclick="openPp('/persons/${pmPerson}/photos/${encodeURIComponent(f)}/img','${f}')"
           onerror="this.style.display='none'">
      <div class="pname">${f}</div>
    </div>`).join('');
}

let ppFile = '';
function openPp(src, fname){
  ppFile = fname;
  document.getElementById('ppImg').src = src;
  document.getElementById('pp').classList.add('on');
}
function closePp(){
  document.getElementById('pp').classList.remove('on');
  ppFile = '';
}
async function deletePpPhoto(){
  const fname = ppFile;
  if(!fname || !confirm(`Видалити ${fname}?`)) return;
  closePp();
  await api(`/persons/${pmPerson}/photos/${encodeURIComponent(fname)}`,{method:'DELETE'});
  await loadPmPhotos();
  loadPersons();
}

async function delPhoto(fname){
  if(!confirm(`Видалити ${fname}?`)) return;
  const r = await api(`/persons/${pmPerson}/photos/${encodeURIComponent(fname)}`,{method:'DELETE'});
  await loadPmPhotos();
  loadPersons();
}

function openEnrollForPerson(){
  closePm();
  const sel = document.getElementById('selPerson');
  if([...sel.options].some(o=>o.value===pmPerson)) sel.value=pmPerson;
  document.getElementById('newNameWrap').style.display='none';
  document.querySelector('.card:nth-child(2)').scrollIntoView({behavior:'smooth'});
}

async function deleteAllSnaps(){
  if(!confirm('Видалити всі знімки?')) return;
  const r = await fetch('/snapshots',{method:'DELETE'});
  const d = await r.json();
  showToast(`🗑 Видалено ${d.removed} знімків`);
  loadSnaps();
}

async function loadLogs(){
  try {
    const d = await api('/logs');
    const lb = document.getElementById('lb');
    lb.textContent = d.lines||'(порожньо)';
    lb.scrollTop = lb.scrollHeight;
  } catch(e){}
}

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

function toggleAuto(){
  const btn=document.getElementById('ab');
  if(autoT){ clearInterval(autoT); autoT=null; btn.textContent='▶ Авто'; btn.style.background='#334155'; }
  else { autoT=setInterval(()=>{loadSnaps();loadLogs();},5000); btn.textContent='⏹ Стоп'; btn.style.background='#dc2626'; }
}

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
      const noPersons = pnames.length === 0;
      return `<div class="uc">
        <img src="/unknown/${encodeURIComponent(f)}/img" onclick="openMo(this.src)" onerror="this.parentElement.remove()">
        <div class="ui">
          <div class="uk">${f}</div>
          <div class="ua">
            <select id="us${i}" onchange="toggleNewUn(${i})">
              ${opts}
              <option value="__new__">+ Нова особа...</option>
            </select>
            <button class="ua ub" onclick="assignUnknown('${f}',${i})">→</button>
            <button class="ua ud" onclick="deleteUnknown('${f}')">✕</button>
          </div>
          <input type="text" id="un${i}" placeholder="ім'я латиницею"
            style="display:${noPersons?'block':'none'};margin-top:5px;width:100%;padding:5px 8px;border-radius:6px;border:1px solid #475569;background:#1e293b;color:#e2e8f0;font-size:.8rem">
        </div>
      </div>`;
    }).join('');
  } catch(e){ console.error('loadUnknown',e); }
}

function toggleNewUn(i){
  const sel = document.getElementById('us'+i);
  const inp = document.getElementById('un'+i);
  inp.style.display = sel.value==='__new__' ? 'block' : 'none';
}

async function assignUnknown(fname, i){
  const sel = document.getElementById('us'+i);
  if(!sel) return;
  let person = sel.value;
  if(person === '__new__'){
    person = (document.getElementById('un'+i).value||'').trim();
    if(!person){ alert("Введіть ім'я"); return; }
  }
  try {
    const r = await fetch(`/unknown/${encodeURIComponent(fname)}/assign?person=${encodeURIComponent(person)}`,{method:'POST'});
    const d = await r.json();
    if(r.ok){
      showToast(d.warning ? `⚠ ${person}: збережено без ембедінгу` : `✓ ${person}: додано`);
      loadUnknown(); loadPersons();
    } else {
      showToast('✗ ' + (d.error||'Помилка'), true);
    }
  } catch(e){ showToast('✗ ' + e.message, true); }
}

function showToast(msg, err){
  let t = document.getElementById('toast');
  if(!t){ t=document.createElement('div'); t.id='toast';
    t.style='position:fixed;bottom:20px;left:50%;transform:translateX(-50%);padding:10px 20px;border-radius:8px;font-size:.85rem;font-weight:600;z-index:999;transition:opacity .3s';
    document.body.appendChild(t); }
  t.textContent=msg; t.style.background=err?'#450a0a':'#14532d';
  t.style.color=err?'#fca5a5':'#86efac'; t.style.opacity='1';
  clearTimeout(t._t); t._t=setTimeout(()=>t.style.opacity='0',3000);
}

async function deleteUnknown(fname){
  if(!confirm(`Видалити ${fname}?`)) return;
  await api(`/unknown/${encodeURIComponent(fname)}`,{method:'DELETE'});
  loadUnknown();
}

checkHealth();
loadPersons();
loadSnaps();
loadLogs();
loadUnknown();
setInterval(checkHealth,15000);
</script>
</body>
</html>"""


def create_app(face_db, snapshot_dir=SNAPSHOT_DIR):
    app = Flask(__name__)
    unknown_dir = os.path.join(os.path.dirname(snapshot_dir), "unknown")
    _rebuild_lock = threading.Lock()

    def _rebuild_bg(name):
        if _rebuild_lock.acquire(blocking=False):
            try:
                face_db.rebuild_person(name)
            finally:
                _rebuild_lock.release()

    @app.route("/")
    def index():
        return Response(HTML, mimetype="text/html; charset=utf-8")

    @app.route("/persons", methods=["GET"])
    def list_persons():
        return jsonify({"persons": face_db.list_persons()})

    @app.route("/persons/<name>", methods=["DELETE"])
    def delete_person(name):
        ok = face_db.delete_person(name)
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
        count = face_db.enroll(name, img)
        if count == 0:
            return jsonify({"error": "no face detected"}), 422
        return jsonify({"status": "enrolled", "name": name, "total": count})

    @app.route("/persons/<name>/photos", methods=["GET"])
    def list_photos(name):
        return jsonify({"name": name, "photos": face_db.photo_count(name)})

    @app.route("/persons/<name>/photos/list", methods=["GET"])
    def list_photo_files(name):
        person_dir = os.path.join(face_db.faces_dir, name)
        if not os.path.isdir(person_dir):
            return jsonify({"error": "not found"}), 404
        files = sorted(f for f in os.listdir(person_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        return jsonify({"name": name, "files": files})

    @app.route("/persons/<name>/photos/<fname>", methods=["DELETE"])
    def delete_photo(name, fname):
        person_dir = os.path.join(face_db.faces_dir, name)
        path = os.path.join(person_dir, fname)
        # safety: must be inside person_dir
        if not os.path.realpath(path).startswith(os.path.realpath(person_dir)):
            return jsonify({"error": "forbidden"}), 403
        if not os.path.isfile(path):
            return jsonify({"error": "not found"}), 404
        os.remove(path)
        threading.Thread(target=_rebuild_bg, args=(name,), daemon=True).start()
        remaining = len([f for f in os.listdir(person_dir)
                         if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        return jsonify({"status": "deleted", "remaining": remaining})

    @app.route("/persons/<name>/photos/<fname>/img", methods=["GET"])
    def photo_img(name, fname):
        person_dir = os.path.join(face_db.faces_dir, name)
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
        count = face_db.enroll(person, img)
        # move file to person folder regardless of enrollment result
        import shutil
        person_dir = os.path.join(face_db.faces_dir, person)
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

    @app.route("/reload", methods=["POST"])
    def reload_db():
        face_db.reload()
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

    return app


def run_api(face_db, host, port, snapshot_dir=SNAPSHOT_DIR):
    flask_app = create_app(face_db, snapshot_dir)
    t = threading.Thread(
        target=lambda: flask_app.run(host=host, port=port, debug=False, use_reloader=False),
        daemon=True,
        name="api",
    )
    t.start()
    log.info("Web UI on http://%s:%s", host, port)
