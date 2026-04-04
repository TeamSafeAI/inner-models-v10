/**
 * view_3d.js — Three.js 3D brain viewer.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { loadVersions, loadBrainList, loadBrainData, assignLayers, updateStats, neuronColorRGB, extent } from './brain_api.js';

// ── State ──
let brainData = null;
let neuronPoints = null;
let connectionLines = null;
let layerMode = 'all';
let currentLayer = 0;
let nLayers = 20;
let layerWidth = 3;
let centerOffset = [0, 0, 0];

// ── Three.js setup ──
const container = document.getElementById('container');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a1a);

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 500);
camera.position.set(20, 15, 25);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
container.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 2;
controls.maxDistance = 200;

scene.add(new THREE.AmbientLight(0xffffff, 0.6));

const raycaster = new THREE.Raycaster();
raycaster.params.Points.threshold = 0.5;
const mouse = new THREE.Vector2();
const tooltip = document.getElementById('tooltip');

// ── Load brain ──
async function onBrainSelected(path) {
    brainData = await loadBrainData(path);
    const info = assignLayers(brainData);
    nLayers = info.nLayers;
    document.getElementById('layerSlider').max = nLayers - 1;
    currentLayer = 0;
    document.getElementById('layerSlider').value = 0;
    document.getElementById('layerVal').textContent = '0';
    buildScene();
    updateStats(brainData);
}

function buildScene() {
    if (neuronPoints) scene.remove(neuronPoints);
    if (connectionLines) scene.remove(connectionLines);

    const d = brainData;
    const n = d.neurons.length;

    const ex = extent(d.neurons.map(n => n.x));
    const ey = extent(d.neurons.map(n => n.y));
    const ez = extent(d.neurons.map(n => n.z));

    const cx = (ex.min + ex.max) / 2;
    const cy = (ey.min + ey.max) / 2;
    const cz = (ez.min + ez.max) / 2;
    centerOffset = [cx, cy, cz];

    const positions = new Float32Array(n * 3);
    const colors = new Float32Array(n * 3);

    for (let i = 0; i < n; i++) {
        positions[i * 3]     = d.neurons[i].x - cx;
        positions[i * 3 + 1] = d.neurons[i].y - cy;
        positions[i * 3 + 2] = d.neurons[i].z - cz;
    }

    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const mat = new THREE.PointsMaterial({
        size: 0.4,
        vertexColors: true,
        sizeAttenuation: true,
        transparent: true,
        opacity: 0.9,
    });

    neuronPoints = new THREE.Points(geom, mat);
    scene.add(neuronPoints);

    buildConnections();
    updateColors();

    // Fit camera to brain extents
    const maxRange = Math.max(ex.range, ey.range, ez.range) || 10;
    camera.position.set(maxRange * 0.8, maxRange * 0.6, maxRange * 1.0);
    controls.target.set(0, 0, 0);
    controls.update();
}

function buildConnections() {
    if (connectionLines) scene.remove(connectionLines);

    const showConn = document.getElementById('showConns').checked;
    if (!showConn || !brainData) return;

    const topN = parseInt(document.getElementById('connCount').value);
    const conns = brainData.connections.slice(0, topN);
    if (conns.length === 0) return;

    const [cx, cy, cz] = centerOffset;
    const positions = new Float32Array(conns.length * 6);
    const colors = new Float32Array(conns.length * 6);

    const ew = extent(conns.map(c => Math.abs(c.w)));
    const minW = ew.min, wRange = ew.range;

    for (let i = 0; i < conns.length; i++) {
        const c = conns[i];
        const src = brainData.neurons[c.src];
        const tgt = brainData.neurons[c.tgt];
        if (!src || !tgt) continue;

        positions[i*6]   = src.x - cx;
        positions[i*6+1] = src.y - cy;
        positions[i*6+2] = src.z - cz;
        positions[i*6+3] = tgt.x - cx;
        positions[i*6+4] = tgt.y - cy;
        positions[i*6+5] = tgt.z - cz;

        const t = (Math.abs(c.w) - minW) / wRange;
        const a = 0.15 + t * 0.6;

        if (c.w > 0) {
            colors[i*6]=1.0*a; colors[i*6+1]=0.55*a; colors[i*6+2]=0.2*a;
            colors[i*6+3]=1.0*a; colors[i*6+4]=0.55*a; colors[i*6+5]=0.2*a;
        } else {
            colors[i*6]=0.3*a; colors[i*6+1]=0.5*a; colors[i*6+2]=1.0*a;
            colors[i*6+3]=0.3*a; colors[i*6+4]=0.5*a; colors[i*6+5]=1.0*a;
        }
    }

    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    connectionLines = new THREE.LineSegments(geom, new THREE.LineBasicMaterial({
        vertexColors: true, transparent: true, opacity: 0.5,
    }));
    scene.add(connectionLines);
}

function updateColors() {
    if (!brainData || !neuronPoints) return;

    const colorMode = document.getElementById('colorMode').value;
    const dotSize = parseFloat(document.getElementById('dotSize').value);
    const colors = neuronPoints.geometry.attributes.color;

    for (let i = 0; i < brainData.neurons.length; i++) {
        const n = brainData.neurons[i];
        let [r, g, b] = neuronColorRGB(n, brainData, colorMode, nLayers);

        // Layer visibility
        if (layerMode === 'filter') {
            const halfW = Math.floor(layerWidth / 2);
            if (Math.abs(n.layer - currentLayer) > halfW) {
                r *= 0.15; g *= 0.15; b *= 0.15;
            }
        }

        colors.setXYZ(i, r, g, b);
    }

    colors.needsUpdate = true;
    neuronPoints.material.size = dotSize * 0.15;
}

// ── Events ──
document.getElementById('brainSelect').addEventListener('change', (e) => {
    if (e.target.value) onBrainSelected(e.target.value);
});
document.getElementById('dotSize').addEventListener('input', (e) => {
    document.getElementById('dotSizeVal').textContent = e.target.value;
    updateColors();
});
document.getElementById('showConns').addEventListener('change', (e) => {
    document.getElementById('connControls').style.display = e.target.checked ? 'flex' : 'none';
    buildConnections();
});
document.getElementById('connCount').addEventListener('input', (e) => {
    document.getElementById('connCountVal').textContent = e.target.value;
    buildConnections();
});
document.getElementById('colorMode').addEventListener('change', updateColors);

function setLayerMode(mode) {
    layerMode = mode;
    const allBtn = document.getElementById('layerAll');
    const filterBtn = document.getElementById('layerFilter');
    const filterCtrl = document.getElementById('layerFilterControls');
    if (mode === 'all') {
        allBtn.classList.add('active');
        filterBtn.classList.remove('active');
        filterCtrl.style.display = 'none';
    } else {
        filterBtn.classList.add('active');
        allBtn.classList.remove('active');
        filterCtrl.style.display = 'block';
    }
    updateColors();
    if (mode === 'filter') showLayerIndicator();
}

function setCurrentLayer(layer) {
    currentLayer = layer;
    document.getElementById('layerSlider').value = currentLayer;
    document.getElementById('layerVal').textContent = currentLayer;
    setLayerMode('filter');
}

// ── Program loading ──
async function loadPrograms() {
    try {
        const resp = await fetch('/api/programs');
        const programs = await resp.json();
        const sel = document.getElementById('programSelect');
        sel.innerHTML = '<option value="">— select program —</option>';
        for (const p of programs) {
            const opt = document.createElement('option');
            opt.value = p.file;
            opt.textContent = p.name;
            opt.dataset.desc = p.description;
            sel.appendChild(opt);
        }
    } catch (e) { /* server not ready */ }
}

document.getElementById('programSelect').addEventListener('change', (e) => {
    const opt = e.target.selectedOptions[0];
    document.getElementById('programDesc').textContent = opt?.dataset?.desc || '';
});

document.getElementById('runBrain').addEventListener('click', async () => {
    const brainPath = document.getElementById('brainSelect').value;
    const version = document.getElementById('versionSelect').value;
    const program = document.getElementById('programSelect').value || null;
    if (!brainPath) {
        document.getElementById('liveStatus').textContent = 'Select a brain first';
        document.getElementById('liveStatus').style.color = '#a44';
        return;
    }
    // Stop any existing simulation
    stopLive();
    try {
        await fetch('/api/stop', { method: 'POST' });
    } catch (e) { /* ok */ }

    // Start simulation on server
    const resp = await fetch('/api/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ brain: brainPath, version, program }),
    });
    const result = await resp.json();
    if (result.error) {
        document.getElementById('liveStatus').textContent = result.error;
        document.getElementById('liveStatus').style.color = '#a44';
        return;
    }
    // Mark Run button active
    document.getElementById('runBrain').classList.add('active');
    document.getElementById('runBrain').textContent = 'Running';
    // Auto-connect live after a brief delay for WS server to start
    document.getElementById('liveStatus').textContent = 'Starting...';
    document.getElementById('liveStatus').style.color = '#aa4';
    setTimeout(() => startLive(), 500);
});

document.getElementById('stopBrain').addEventListener('click', async () => {
    stopLive();
    try {
        await fetch('/api/stop', { method: 'POST' });
    } catch (e) { /* ok */ }
    document.getElementById('runBrain').classList.remove('active');
    document.getElementById('runBrain').textContent = 'Run';
    document.getElementById('liveStatus').textContent = 'Stopped';
    document.getElementById('liveStatus').style.color = '#666';
});

document.getElementById('layerAll').addEventListener('click', () => setLayerMode('all'));
document.getElementById('layerFilter').addEventListener('click', () => setLayerMode('filter'));
document.getElementById('layerSlider').addEventListener('input', (e) => {
    currentLayer = parseInt(e.target.value);
    document.getElementById('layerVal').textContent = currentLayer;
    updateColors();
    showLayerIndicator();
});
document.getElementById('layerWidth').addEventListener('input', (e) => {
    layerWidth = parseInt(e.target.value);
    document.getElementById('layerWidthVal').textContent = layerWidth;
    updateColors();
});

document.addEventListener('keydown', (e) => {
    if ((e.key === 'ArrowUp' || e.key === 'ArrowRight') && currentLayer < nLayers - 1) {
        setCurrentLayer(currentLayer + 1);
    } else if ((e.key === 'ArrowDown' || e.key === 'ArrowLeft') && currentLayer > 0) {
        setCurrentLayer(currentLayer - 1);
    } else if (e.key === 'Escape') {
        setLayerMode('all');
    }
});

let layerTimeout;
function showLayerIndicator() {
    if (!brainData) return;
    const ind = document.getElementById('layerIndicator');
    const halfW = Math.floor(layerWidth / 2);
    const count = brainData.neurons.filter(n => Math.abs(n.layer - currentLayer) <= halfW).length;
    ind.innerHTML = `Layer <b>${currentLayer}</b> / ${nLayers - 1} &nbsp;·&nbsp; ${count} neurons`;
    ind.style.opacity = '1';
    clearTimeout(layerTimeout);
    layerTimeout = setTimeout(() => { ind.style.opacity = '0'; }, 2000);
}

// Hover
renderer.domElement.addEventListener('mousemove', (e) => {
    if (!brainData || !neuronPoints) return;
    mouse.x = ((e.clientX - 260) / (window.innerWidth - 260)) * 2 - 1;
    mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const hits = raycaster.intersectObject(neuronPoints);

    if (hits.length > 0) {
        const idx = hits[0].index;
        const n = brainData.neurons[idx];
        let text = `#${n.id}`;
        const name = brainData.neuron_names[String(idx)];
        if (name) text += `  ${name}`;
        text += `\nPos: (${n.x.toFixed(1)}, ${n.y.toFixed(1)}, ${n.z.toFixed(1)})`;
        text += `\nLayer: ${n.layer}`;
        text += `\n${n.type === 0 ? 'Excitatory' : 'Inhibitory'}`;
        const func = brainData.neuron_types[String(idx)];
        if (func) text += ` (${func})`;

        tooltip.textContent = text;
        tooltip.style.display = 'block';
        tooltip.style.left = (e.clientX + 15) + 'px';
        tooltip.style.top = (e.clientY - 10) + 'px';
    } else {
        tooltip.style.display = 'none';
    }
});

// Resize
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

// ── Live spike visualization via WebSocket ──
let liveMode = false;
let liveWs = null;
let liveReconnect = null;
let spikeBrightness = null;  // per-neuron brightness (0-1), decays each frame
const SPIKE_DECAY = 0.92;    // how fast spikes fade
const SPIKE_BRIGHTNESS = 1.0;
const WS_URL = 'ws://localhost:8878';

function startLive() {
    if (liveMode) return;
    liveMode = true;
    spikeBrightness = new Float32Array(brainData ? brainData.neurons.length : 0);

    const status = document.getElementById('liveStatus');
    status.textContent = 'Connecting...';
    status.style.color = '#aa4';

    connectWs();
}

function stopLive() {
    liveMode = false;
    if (liveReconnect) { clearTimeout(liveReconnect); liveReconnect = null; }
    if (liveWs) { liveWs.close(); liveWs = null; }
    spikeBrightness = null;
    arenaTrail.length = 0;
    if (brainData) updateColors();
}

function connectWs() {
    if (!liveMode) return;
    const status = document.getElementById('liveStatus');

    try {
        liveWs = new WebSocket(WS_URL);
    } catch (e) {
        status.textContent = 'Connection failed';
        status.style.color = '#a44';
        scheduleReconnect();
        return;
    }

    liveWs.onopen = () => {
        status.textContent = 'Live';
        status.style.color = '#4a4';
    };

    liveWs.onmessage = (event) => {
        if (!brainData) return;
        const data = JSON.parse(event.data);

        // Initialize brightness array if needed
        if (!spikeBrightness || spikeBrightness.length !== brainData.neurons.length) {
            spikeBrightness = new Float32Array(brainData.neurons.length);
        }

        // Set spiked neurons to bright
        for (const id of data.spikes) {
            if (id < spikeBrightness.length) {
                spikeBrightness[id] = SPIKE_BRIGHTNESS;
            }
        }

        // Update status overlay
        const ind = document.getElementById('layerIndicator');
        ind.innerHTML = `<b>LIVE</b> tick ${data.tick} · ${data.n_fired}/${data.n_total} fired`;
        ind.style.opacity = '1';

        // Arena overlay
        if (data.arena) {
            updateArenaOverlay(data.arena);
        }

        if (!data.running) {
            ind.innerHTML += ' · <span style="color:#f66">stopped</span>';
            stopLive();
            document.getElementById('runBrain').classList.remove('active');
            document.getElementById('runBrain').textContent = 'Run';
            document.getElementById('liveStatus').textContent = 'Finished';
            document.getElementById('liveStatus').style.color = '#888';
            // Hide arena after stop
            if (arenaCanvas) arenaCanvas.style.display = 'none';
            if (arenaStatsEl) arenaStatsEl.style.display = 'none';
        }
    };

    liveWs.onclose = () => {
        if (liveMode) {
            status.textContent = 'Reconnecting...';
            status.style.color = '#aa4';
            scheduleReconnect();
        }
    };

    liveWs.onerror = () => {
        // onclose will fire after this
    };
}

function scheduleReconnect() {
    if (!liveMode || liveReconnect) return;
    liveReconnect = setTimeout(() => {
        liveReconnect = null;
        connectWs();
    }, 1000);
}

function updateLiveColors() {
    if (!brainData || !neuronPoints || !spikeBrightness) return;

    const colors = neuronPoints.geometry.attributes.color;
    const colorMode = document.getElementById('colorMode').value;

    for (let i = 0; i < brainData.neurons.length; i++) {
        const n = brainData.neurons[i];
        let [r, g, b] = neuronColorRGB(n, brainData, colorMode, nLayers);

        // Blend toward white based on spike brightness
        const s = spikeBrightness[i];
        if (s > 0.01) {
            r = r + (1.0 - r) * s;
            g = g + (1.0 - g) * s;
            b = b + (1.0 - b) * s;
        } else {
            // Dim non-firing neurons slightly
            r *= 0.4; g *= 0.4; b *= 0.4;
        }

        // Layer filter still works in live mode
        if (layerMode === 'filter') {
            const halfW = Math.floor(layerWidth / 2);
            if (Math.abs(n.layer - currentLayer) > halfW) {
                r *= 0.15; g *= 0.15; b *= 0.15;
            }
        }

        colors.setXYZ(i, r, g, b);
    }

    // Decay all brightness
    for (let i = 0; i < spikeBrightness.length; i++) {
        spikeBrightness[i] *= SPIKE_DECAY;
    }

    colors.needsUpdate = true;
}

// ── Arena overlay ──
const arenaCanvas = document.getElementById('arenaCanvas');
const arenaCtx = arenaCanvas ? arenaCanvas.getContext('2d') : null;
const arenaStatsEl = document.getElementById('arenaStats');
const arenaTrail = [];  // recent positions for trail
const TRAIL_MAX = 200;

function updateArenaOverlay(arena) {
    if (!arenaCtx || !arenaCanvas) return;

    // Show canvas + stats on first data
    arenaCanvas.style.display = 'block';
    if (arenaStatsEl) arenaStatsEl.style.display = 'block';

    const W = arenaCanvas.width;
    const H = arenaCanvas.height;
    const scale = W / arena.size;

    arenaCtx.clearRect(0, 0, W, H);

    // Background
    arenaCtx.fillStyle = '#0d0d1a';
    arenaCtx.fillRect(0, 0, W, H);

    // Grid lines
    arenaCtx.strokeStyle = '#1a1a2e';
    arenaCtx.lineWidth = 0.5;
    for (let g = 0; g <= arena.size; g += 20) {
        const gx = g * scale;
        arenaCtx.beginPath(); arenaCtx.moveTo(gx, 0); arenaCtx.lineTo(gx, H); arenaCtx.stroke();
        arenaCtx.beginPath(); arenaCtx.moveTo(0, gx); arenaCtx.lineTo(W, gx); arenaCtx.stroke();
    }

    // Heat patches (orange circles)
    if (arena.heat_patches) {
        for (const [hx, hy, hr, hi] of arena.heat_patches) {
            const sx = hx * scale, sy = hy * scale, sr = hr * scale;
            const grad = arenaCtx.createRadialGradient(sx, sy, 0, sx, sy, sr);
            grad.addColorStop(0, `rgba(255, 100, 30, ${0.3 * hi})`);
            grad.addColorStop(1, 'rgba(255, 100, 30, 0)');
            arenaCtx.fillStyle = grad;
            arenaCtx.beginPath();
            arenaCtx.arc(sx, sy, sr, 0, Math.PI * 2);
            arenaCtx.fill();
        }
    }

    // Food zone (green circle)
    if (arena.food_center) {
        const fx = arena.food_center[0] * scale;
        const fy = arena.food_center[1] * scale;
        const fr = arena.food_radius * scale;
        const grad = arenaCtx.createRadialGradient(fx, fy, 0, fx, fy, fr);
        grad.addColorStop(0, 'rgba(50, 220, 80, 0.25)');
        grad.addColorStop(1, 'rgba(50, 220, 80, 0.05)');
        arenaCtx.fillStyle = grad;
        arenaCtx.beginPath();
        arenaCtx.arc(fx, fy, fr, 0, Math.PI * 2);
        arenaCtx.fill();
        arenaCtx.strokeStyle = 'rgba(50, 220, 80, 0.4)';
        arenaCtx.lineWidth = 1;
        arenaCtx.stroke();
    }

    // Trail
    arenaTrail.push({ x: arena.x, y: arena.y, pain: arena.pain });
    if (arenaTrail.length > TRAIL_MAX) arenaTrail.shift();

    for (let i = 0; i < arenaTrail.length - 1; i++) {
        const t = arenaTrail[i];
        const alpha = (i / arenaTrail.length) * 0.4;
        arenaCtx.fillStyle = t.pain
            ? `rgba(255, 80, 80, ${alpha})`
            : `rgba(100, 160, 255, ${alpha})`;
        arenaCtx.fillRect(t.x * scale - 0.5, t.y * scale - 0.5, 1.5, 1.5);
    }

    // Body dot (current position)
    const bx = arena.x * scale;
    const by = arena.y * scale;
    arenaCtx.beginPath();
    arenaCtx.arc(bx, by, 4, 0, Math.PI * 2);
    arenaCtx.fillStyle = arena.pain ? '#ff4444' : '#ffffff';
    arenaCtx.fill();
    arenaCtx.strokeStyle = arena.pain ? '#ff8888' : '#88aaff';
    arenaCtx.lineWidth = 1.5;
    arenaCtx.stroke();

    // Heading arrow (when body mode sends heading)
    if (arena.heading !== null && arena.heading !== undefined) {
        const h = arena.heading;
        const arrowLen = 12;
        const ax = bx + Math.cos(h) * arrowLen;
        const ay = by + Math.sin(h) * arrowLen;
        arenaCtx.beginPath();
        arenaCtx.moveTo(bx, by);
        arenaCtx.lineTo(ax, ay);
        arenaCtx.strokeStyle = arena.pain ? '#ff8888' : '#88ccff';
        arenaCtx.lineWidth = 2;
        arenaCtx.stroke();
        // Arrowhead
        const headLen = 4;
        const ha1 = h + 2.5;
        const ha2 = h - 2.5;
        arenaCtx.beginPath();
        arenaCtx.moveTo(ax, ay);
        arenaCtx.lineTo(ax - Math.cos(ha1) * headLen, ay - Math.sin(ha1) * headLen);
        arenaCtx.moveTo(ax, ay);
        arenaCtx.lineTo(ax - Math.cos(ha2) * headLen, ay - Math.sin(ha2) * headLen);
        arenaCtx.stroke();
    }

    // Stats text
    if (arenaStatsEl) {
        const painColor = arena.pain ? '#f66' : '#6a6';
        let headingStr = '';
        if (arena.heading !== null && arena.heading !== undefined) {
            const deg = Math.round(arena.heading * 180 / Math.PI) % 360;
            headingStr = `Hdg <b>${deg}</b>&deg; &nbsp; `;
        }
        arenaStatsEl.innerHTML =
            `Ep <b>${arena.episode}</b> &nbsp; ` +
            headingStr +
            `Dist <b>${arena.dist}</b> &nbsp; ` +
            `Food <b style="color:#4a4">${arena.food.toFixed(2)}</b> &nbsp; ` +
            `<span style="color:${painColor}">${arena.pain ? 'PAIN' : 'calm'}</span>`;
    }
}

// Animate
function animate() {
    requestAnimationFrame(animate);
    controls.update();
    if (liveMode && spikeBrightness) {
        updateLiveColors();
    }
    renderer.render(scene, camera);
}

// Version selector
const versionSelect = document.getElementById('versionSelect');
const brainSelect = document.getElementById('brainSelect');

loadVersions(versionSelect).then(() => {
    loadBrainList(brainSelect, versionSelect.value);
});
loadPrograms();

versionSelect.addEventListener('change', () => {
    loadBrainList(brainSelect, versionSelect.value);
});

animate();
