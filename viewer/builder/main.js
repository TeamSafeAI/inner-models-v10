/** main.js — V8 Brain Builder. Neuron types on a 3D grid with typed paths. */
import * as THREE from 'three';
import * as Scene from './scene.js';

const NODE_COLORS = { RS: 0x4ade80, FS: 0xef4444, IB: 0xfb923c, CH: 0xfbbf24, LTS: 0x94a3b8 };
const PATH_COLORS = { fixed: 0xcccccc, plastic: 0x60a5fa, facilitating: 0x34d399, depressing: 0xfb923c, gated: 0xc084fc };
const PATH_EXTRA = { plastic: { learning_rate: 0.01 }, facilitating: { facil_increment: 0.1 },
    depressing: { depress_factor: 0.9 }, gated: { gate_threshold: 0.5 } };
const SHORT = { learning_rate: 'LR', facil_increment: 'Facil', depress_factor: 'Depr', gate_threshold: 'Gate' };
const R = 0.35; // sphere radius

let nodeTool = null, pathTool = null, selNode = null, selPath = null, connectFrom = null;
let nodes = [], paths = [], nid = 1, pid = 1;
let dragging = false, dragNode = null;

const wrap = document.getElementById('canvas-wrap');
const helpEl = document.getElementById('helpText');
const infoEl = document.getElementById('selectedInfo');
const statusEl = document.getElementById('genStatus');
const $ = id => document.getElementById(id);

const { scene } = Scene.init(wrap);

// Palette buttons
document.querySelectorAll('.node-btn').forEach(b => b.addEventListener('click', () => {
    const t = b.dataset.type;
    document.querySelectorAll('.node-btn,.path-btn').forEach(x => x.classList.remove('selected'));
    if (nodeTool === t) { nodeTool = null; } else { nodeTool = t; pathTool = null; connectFrom = null; b.classList.add('selected'); }
    updateHelp();
}));
document.querySelectorAll('.path-btn').forEach(b => b.addEventListener('click', () => {
    const t = b.dataset.type;
    document.querySelectorAll('.node-btn,.path-btn').forEach(x => x.classList.remove('selected'));
    if (pathTool === t) { pathTool = null; } else { pathTool = t; nodeTool = null; connectFrom = null; b.classList.add('selected'); }
    updateHelp();
}));

// Config sliders
function bindSlider(id, valId, cb) {
    const s = $(id), v = $(valId);
    s.addEventListener('input', () => { v.textContent = s.dataset.fmt === 'f' ? parseFloat(s.value).toFixed(1) : s.value; cb(s.value); });
}
bindSlider('cfgNeurons', 'cfgNeuronsVal', v => { if (selNode) selNode.neurons = +v; });
bindSlider('cfgWeight', 'cfgWeightVal', v => { if (selPath) selPath.weight = +v; });
$('cfgWeight').dataset.fmt = 'f';
bindSlider('cfgDelay', 'cfgDelayVal', v => { if (selPath) selPath.delay = +v; });

// Action buttons
$('btnSave').addEventListener('click', saveLayout);
$('btnLoad').addEventListener('click', loadFromFile);
$('btnGenerate').addEventListener('click', () => generate(false));
$('btnGenerateRun').addEventListener('click', () => generate(true));
$('btnClear').addEventListener('click', clearAll);
refreshSavedList();

// Canvas events
wrap.addEventListener('mousedown', e => { if (e.button === 0) onDown(e); });
wrap.addEventListener('mousemove', e => { if (dragNode) onDrag(e); });
wrap.addEventListener('mouseup', () => { if (dragNode) Scene.getControls().enabled = true; dragging = false; dragNode = null; });
wrap.addEventListener('contextmenu', e => { e.preventDefault(); onRight(e); });

function hitNode(e) { const m = Scene.raycastMeshes(e, wrap, nodes.map(n => n.mesh)); return m ? nodes.find(n => n.mesh === m) : null; }
function hitPath(e) { const m = Scene.raycastMeshes(e, wrap, paths.map(p => p.tube)); return m ? paths.find(p => p.tube === m) : null; }

function onDown(e) {
    const node = hitNode(e);
    if (node) {
        if (e.shiftKey && pathTool) {
            if (!connectFrom) { connectFrom = node; node.mesh.material.emissiveIntensity = 0.8; helpEl.textContent = `FROM ${node.type}#${node.id} — Shift+click target`; }
            else if (connectFrom.id !== node.id) { createPath(connectFrom, node, pathTool); connectFrom.mesh.material.emissiveIntensity = 0.3; connectFrom = null; }
            return;
        }
        dragNode = node; dragging = false; selectNode(node); return;
    }
    const p = hitPath(e);
    if (p) { selectPath(p); return; }
    if (nodeTool && !e.shiftKey) {
        const g = Scene.raycastGrid(e, wrap);
        if (g && !nodes.find(n => n.x === g.x && n.z === g.z)) placeNode(nodeTool, g.x, g.z);
        return;
    }
    deselectAll();
}

function onDrag(e) {
    dragging = true; Scene.getControls().enabled = false;
    const g = Scene.raycastGrid(e, wrap);
    if (!g) return;
    dragNode.x = g.x; dragNode.z = g.z;
    dragNode.mesh.position.set(g.x, R, g.z);
    dragNode.label.position.set(g.x, R * 2 + 0.6, g.z);
    updatePathLines(dragNode);
}

function onRight(e) {
    const node = hitNode(e); if (node) { removeNode(node); return; }
    const p = hitPath(e); if (p) removePath(p);
}

// --- Node ops ---
function placeNode(type, x, z) {
    const c = NODE_COLORS[type];
    const mesh = new THREE.Mesh(new THREE.SphereGeometry(R, 16, 16), new THREE.MeshPhongMaterial({ color: c, emissive: c, emissiveIntensity: 0.3 }));
    mesh.position.set(x, R, z); scene.add(mesh);
    const label = makeLabel(type, c); label.position.set(x, R * 2 + 0.6, z); scene.add(label);
    const node = { id: nid++, type, x, z, neurons: 50, mesh, label };
    nodes.push(node); selectNode(node);
}

function removeNode(node) {
    paths.filter(p => p.from === node.id || p.to === node.id).forEach(p => removePath(p));
    scene.remove(node.mesh); scene.remove(node.label);
    node.mesh.geometry.dispose(); node.mesh.material.dispose();
    nodes = nodes.filter(n => n.id !== node.id);
    if (selNode?.id === node.id) deselectAll();
}

// --- Path ops ---
function createPath(fromN, toN, type) {
    if (paths.find(p => (p.from === fromN.id && p.to === toN.id) || (p.from === toN.id && p.to === fromN.id))) return;
    const c = PATH_COLORS[type];
    const pts = [new THREE.Vector3(fromN.x, R, fromN.z), new THREE.Vector3(toN.x, R, toN.z)];
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: c }));
    scene.add(line);
    const len = new THREE.Vector3(toN.x - fromN.x, 0, toN.z - fromN.z).length();
    const tube = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, len, 4), new THREE.MeshBasicMaterial({ color: c, transparent: true, opacity: 0.3 }));
    tube.position.set((fromN.x + toN.x) / 2, R, (fromN.z + toN.z) / 2);
    tube.lookAt(new THREE.Vector3(toN.x, R, toN.z)); tube.rotateX(Math.PI / 2);
    scene.add(tube);
    const params = PATH_EXTRA[type] ? { ...PATH_EXTRA[type] } : {};
    const path = { id: pid++, from: fromN.id, to: toN.id, pathType: type, weight: 1.0, delay: 1, params, line, tube };
    paths.push(path); selectPath(path);
}

function removePath(p) {
    scene.remove(p.line); scene.remove(p.tube);
    p.line.geometry.dispose(); p.line.material.dispose();
    p.tube.geometry.dispose(); p.tube.material.dispose();
    paths = paths.filter(x => x.id !== p.id);
    if (selPath?.id === p.id) deselectAll();
}

function updatePathLines(node) {
    paths.forEach(p => {
        if (p.from !== node.id && p.to !== node.id) return;
        const f = nodes.find(n => n.id === p.from), t = nodes.find(n => n.id === p.to);
        if (!f || !t) return;
        p.line.geometry.dispose();
        p.line.geometry = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(f.x, R, f.z), new THREE.Vector3(t.x, R, t.z)]);
        const len = new THREE.Vector3(t.x - f.x, 0, t.z - f.z).length();
        p.tube.geometry.dispose();
        p.tube.geometry = new THREE.CylinderGeometry(0.06, 0.06, len, 4);
        p.tube.position.set((f.x + t.x) / 2, R, (f.z + t.z) / 2);
        p.tube.rotation.set(0, 0, 0);
        p.tube.lookAt(new THREE.Vector3(t.x, R, t.z)); p.tube.rotateX(Math.PI / 2);
    });
}

// --- Selection ---
function selectNode(node) {
    deselectAll(); selNode = node; node.mesh.material.emissiveIntensity = 0.6;
    infoEl.textContent = `${node.type} #${node.id} at (${node.x}, ${node.z})`;
    $('nodeConfigInfo').style.display = 'none';
    $('nodeConfigControls').style.display = 'block';
    $('cfgNodeType').textContent = node.type;
    $('cfgNodePos').textContent = `(${node.x}, ${node.z})`;
    $('cfgNeurons').value = node.neurons; $('cfgNeuronsVal').textContent = node.neurons;
}

function selectPath(path) {
    deselectAll(); selPath = path;
    path.line.material.color.set(0xffffff); path.tube.material.opacity = 0.6;
    const f = nodes.find(n => n.id === path.from), t = nodes.find(n => n.id === path.to);
    infoEl.textContent = `${path.pathType}: ${f?.type}#${path.from} -> ${t?.type}#${path.to}`;
    $('pathConfigInfo').style.display = 'none';
    $('pathConfigControls').style.display = 'block';
    $('cfgPathType').textContent = path.pathType;
    $('cfgWeight').value = path.weight; $('cfgWeightVal').textContent = path.weight.toFixed(1);
    $('cfgDelay').value = path.delay; $('cfgDelayVal').textContent = path.delay;
    const extra = $('pathExtraParams'); extra.innerHTML = '';
    for (const [key, val] of Object.entries(path.params)) {
        const row = document.createElement('div'); row.className = 'control-row';
        const lbl = document.createElement('label'); lbl.textContent = SHORT[key] || key;
        const inp = document.createElement('input'); Object.assign(inp, { type: 'range', min: '0.001', max: '1.0', step: '0.01', value: val }); inp.style.flex = '1';
        const sp = document.createElement('span'); sp.className = 'val'; sp.textContent = val;
        inp.addEventListener('input', () => { path.params[key] = +inp.value; sp.textContent = (+inp.value).toFixed(2); });
        row.append(lbl, inp, sp); extra.appendChild(row);
    }
}

function deselectAll() {
    if (selNode) { selNode.mesh.material.emissiveIntensity = 0.3; selNode = null; }
    if (selPath) { selPath.line.material.color.set(PATH_COLORS[selPath.pathType]); selPath.tube.material.opacity = 0.3; selPath = null; }
    infoEl.textContent = 'Click a node or path to select';
    $('nodeConfigInfo').style.display = 'block'; $('nodeConfigControls').style.display = 'none';
    $('pathConfigInfo').style.display = 'block'; $('pathConfigControls').style.display = 'none';
    $('pathExtraParams').innerHTML = '';
}

// --- Export / Import ---
function exportLayout() {
    return {
        nodes: nodes.map(n => ({ id: n.id, type: n.type, x: n.x, z: n.z, neurons: n.neurons })),
        paths: paths.map(p => ({ from: p.from, to: p.to, pathType: p.pathType, weight: p.weight, delay: p.delay, params: { ...p.params } })),
    };
}

function importLayout(data) {
    clearAll(); if (!data.nodes || !data.paths) return;
    data.nodes.forEach(nd => { placeNode(nd.type, nd.x, nd.z); nodes[nodes.length - 1].neurons = nd.neurons || 50; });
    const idMap = {}; data.nodes.forEach((nd, i) => { idMap[nd.id] = nodes[i].id; });
    data.paths.forEach(pd => {
        const f = nodes.find(n => n.id === idMap[pd.from]), t = nodes.find(n => n.id === idMap[pd.to]);
        if (f && t) { createPath(f, t, pd.pathType); const p = paths[paths.length - 1]; if (p) { p.weight = pd.weight || 1; p.delay = pd.delay || 1; if (pd.params) p.params = { ...pd.params }; } }
    });
    deselectAll();
}

// --- Save / Load / Generate ---
async function saveLayout() {
    const name = $('layoutName').value.trim(); if (!name) return;
    try {
        const r = await fetch('/api/build/save', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ tier: 'sub_components', name, data: exportLayout() }) });
        const j = await r.json(); statusEl.textContent = j.ok ? `Saved "${name}"` : (j.error || 'Failed'); refreshSavedList();
    } catch (e) { statusEl.textContent = 'Save error: ' + e.message; }
}

function loadFromFile() {
    const inp = document.createElement('input'); inp.type = 'file'; inp.accept = '.json';
    inp.onchange = e => { const f = e.target.files[0]; if (!f) return; const r = new FileReader();
        r.onload = ev => { try { const d = JSON.parse(ev.target.result); importLayout(d.data || d); } catch (e) { statusEl.textContent = 'Load error: ' + e.message; } };
        r.readAsText(f); }; inp.click();
}

async function refreshSavedList() {
    const el = $('savedList');
    try {
        const r = await fetch('/api/build/list?tier=sub_components'), resp = await r.json();
        const items = resp.items || resp; el.innerHTML = '';
        if (Array.isArray(items)) items.forEach(item => {
            const name = item.name || item;
            const d = document.createElement('div'); d.className = 'saved-item';
            d.innerHTML = `<span class="name">${name}</span><button class="load-btn">Load</button>`;
            d.querySelector('.load-btn').addEventListener('click', async () => {
                try { const r2 = await fetch(`/api/build/load?tier=sub_components&name=${encodeURIComponent(name)}`);
                    importLayout((await r2.json()).data || await r2.json()); } catch (e) { statusEl.textContent = 'Load error: ' + e.message; }
            }); el.appendChild(d);
        });
    } catch { el.innerHTML = '<div class="info-box">No saved layouts</div>'; }
}

async function generate(run) {
    const name = $('brainName').value.trim(); if (!name) return;
    statusEl.textContent = 'Generating...';
    try {
        const r = await fetch('/api/build/generate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name, layout: exportLayout(), run }) });
        const j = await r.json(); statusEl.textContent = j.error ? j.error : `Generated "${name}" (${j.n_neurons || '?'} neurons, ${j.n_synapses || '?'} synapses)`;
    } catch (e) { statusEl.textContent = 'Error: ' + e.message; }
}

function clearAll() { [...paths].forEach(removePath); [...nodes].forEach(removeNode); deselectAll(); }

// --- Helpers ---
function makeLabel(text, color) {
    const c = document.createElement('canvas'); c.width = 64; c.height = 32;
    const ctx = c.getContext('2d'); ctx.fillStyle = '#' + new THREE.Color(color).getHexString();
    ctx.font = 'bold 20px Segoe UI'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(text, 32, 16);
    const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), transparent: true }));
    s.scale.set(1.2, 0.6, 1); return s;
}

function updateHelp() {
    helpEl.textContent = nodeTool ? `Place ${nodeTool}: click empty grid` : pathTool ? `Create ${pathTool} path: Shift+click source, then target` :
        'Click node type \u2192 click grid to place \u00b7 Click path type \u2192 Shift+click two nodes \u00b7 Right-click: delete \u00b7 Drag: move';
}
