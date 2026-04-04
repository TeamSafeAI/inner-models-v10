/**
 * view_2d.js — 2D canvas layer-slice brain viewer.
 */
import { loadVersions, loadBrainList, loadBrainData, assignLayers, updateStats, neuronColorCSS, extent } from './brain_api.js';

// ── State ──
let brainData = null;
let nLayers = 20;
let currentLayer = 0;
let hoveredNeuron = null;

// Canvas
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const histCanvas = document.getElementById('layerHist');
const histCtx = histCanvas.getContext('2d');
const tooltip = document.getElementById('tooltip');
const legend = document.getElementById('legend');

let W, H;
let offsetX = 0, offsetY = 0;
let scale = 1;
let dragging = false;
let dragStartX, dragStartY;

// ── Load brain ──
async function onBrainSelected(path) {
    brainData = await loadBrainData(path);
    const info = assignLayers(brainData);
    nLayers = info.nLayers;
    currentLayer = 0;
    document.getElementById('layerSlider').max = nLayers - 1;
    document.getElementById('layerSlider').value = 0;
    document.getElementById('layerMax').textContent = nLayers - 1;
    updateStats(brainData);
    fitView();
    updateLayerInfo();
    drawHist();
    draw();
}

function getLayerNeurons() {
    if (!brainData) return [];
    return brainData.neurons.filter(n => n.layer === currentLayer);
}

function worldToScreen(wx, wy) {
    return [wx * scale + offsetX, wy * scale + offsetY];
}
function screenToWorld(sx, sy) {
    return [(sx - offsetX) / scale, (sy - offsetY) / scale];
}

function fitView() {
    const neurons = getLayerNeurons();
    if (neurons.length === 0) return;

    const ey = extent(neurons.map(n => n.y));

    const pad = 120;
    const canvasH = H - 100; // bottom bar
    const canvasW = W - 280; // sidebar

    // Spread neurons horizontally by Y position, vertically by index
    scale = Math.min((canvasW - pad) / ey.range, (canvasH - pad) / (neurons.length * 0.5)) * 0.8;
    scale = Math.max(scale, 5);
    offsetX = 280 + canvasW / 2 - (ey.min + ey.range / 2) * scale;
    offsetY = canvasH / 2 - (neurons.length * 0.25) * scale;
}

function drawHist() {
    if (!brainData) return;
    const hw = histCanvas.width;
    const hh = histCanvas.height;
    histCtx.clearRect(0, 0, hw, hh);

    // Pre-compute counts per layer (single pass)
    const counts = new Array(nLayers).fill(0);
    for (const n of brainData.neurons) counts[n.layer]++;
    let maxCount = 0;
    for (let i = 0; i < nLayers; i++) if (counts[i] > maxCount) maxCount = counts[i];

    const barW = hw / nLayers;

    for (let i = 0; i < nLayers; i++) {
        const barH = (counts[i] / maxCount) * (hh - 6);
        const active = i === currentLayer;

        histCtx.fillStyle = active ? '#4a90d9' : 'rgba(255,255,255,0.15)';
        histCtx.fillRect(i * barW + 1, hh - barH - 2, barW - 2, barH);

        if (nLayers <= 30) {
            histCtx.fillStyle = active ? '#fff' : '#555';
            histCtx.font = '9px Consolas';
            histCtx.textAlign = 'center';
            histCtx.fillText(i, i * barW + barW / 2, hh - 1);
        }
    }
}

function draw() {
    ctx.fillStyle = '#0a0a1a';
    ctx.fillRect(0, 0, W, H);

    if (!brainData) return;

    const neurons = getLayerNeurons();
    const showConn = document.getElementById('showConns').checked;
    const showLabels = document.getElementById('showLabels').checked;
    const colorMode = document.getElementById('colorMode').value;
    const dotSize = parseInt(document.getElementById('dotSize').value);

    const neuronIds = new Set(neurons.map(n => n.id));

    // Screen positions for this layer's neurons
    const screenPos = {};
    neurons.forEach((n, idx) => {
        const [sx, sy] = worldToScreen(n.y, idx * 0.5);
        screenPos[n.id] = [sx, sy];
    });

    // Connections
    if (showConn && brainData.connections) {
        const topN = parseInt(document.getElementById('connCount').value);
        const conns = brainData.connections.slice(0, topN);

        for (const c of conns) {
            const srcIn = neuronIds.has(c.src);
            const tgtIn = neuronIds.has(c.tgt);
            if (!srcIn && !tgtIn) continue;

            let x1, y1, x2, y2;
            if (srcIn) [x1, y1] = screenPos[c.src];
            if (tgtIn) [x2, y2] = screenPos[c.tgt];
            if (!srcIn || !tgtIn) continue; // only draw within-layer for clarity

            const alpha = 0.2;
            ctx.strokeStyle = c.w > 0
                ? `rgba(255, 140, 50, ${alpha})`
                : `rgba(100, 150, 255, ${alpha})`;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
        }
    }

    // Neurons
    for (const n of neurons) {
        const [sx, sy] = screenPos[n.id];
        const r = n === hoveredNeuron ? dotSize + 3 : dotSize;
        const color = neuronColorCSS(n, brainData, colorMode, nLayers);

        ctx.fillStyle = color;
        ctx.globalAlpha = n === hoveredNeuron ? 1.0 : 0.85;
        ctx.beginPath();

        if (n.type === 1) {
            // Triangle for inhibitory
            ctx.moveTo(sx, sy - r);
            ctx.lineTo(sx - r * 0.87, sy + r * 0.5);
            ctx.lineTo(sx + r * 0.87, sy + r * 0.5);
            ctx.closePath();
        } else {
            ctx.arc(sx, sy, r, 0, Math.PI * 2);
        }
        ctx.fill();

        if (n === hoveredNeuron) {
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 2;
            ctx.stroke();
        }

        // Labels
        if (showLabels) {
            const name = brainData.neuron_names[String(n.id)];
            if (name) {
                ctx.fillStyle = '#ccc';
                ctx.font = '10px Consolas';
                ctx.globalAlpha = 0.8;
                ctx.fillText(name, sx + r + 4, sy + 3);
            }
        }
        ctx.globalAlpha = 1.0;
    }

    // Layer title
    ctx.fillStyle = 'rgba(127,170,255,0.4)';
    ctx.font = '14px Segoe UI';
    ctx.fillText(`Layer ${currentLayer} — ${neurons.length} neurons`, 275, 30);

    updateLegend(colorMode);
}

function updateLegend(mode) {
    let html = '';
    if (mode === 'ei') {
        html = `
            <div class="legend-item"><span class="legend-dot" style="background:rgb(74,144,217)"></span> Exc (${brainData.n_exc})</div>
            <div class="legend-item"><span class="legend-dot" style="background:rgb(217,74,74)"></span> Inh (${brainData.n_inh})</div>
        `;
    } else if (mode === 'func') {
        const counts = {};
        for (const n of brainData.neurons) {
            const f = brainData.neuron_types[String(n.id)] || 'unknown';
            counts[f] = (counts[f] || 0) + 1;
        }
        const colorMap = {sensory:'rgb(76,175,80)', motor:'rgb(255,152,0)', command:'rgb(156,39,176)', inter:'rgb(128,128,128)'};
        for (const [f, c] of Object.entries(counts)) {
            html += `<div class="legend-item"><span class="legend-dot" style="background:${colorMap[f]||'#888'}"></span> ${f} (${c})</div>`;
        }
    } else {
        html = '<div class="dim">Color = layer depth</div>';
    }
    legend.innerHTML = html;
}

function updateLayerInfo() {
    if (!brainData) return;
    const count = brainData.neurons.filter(n => n.layer === currentLayer).length;
    document.getElementById('layerNum').textContent = currentLayer;
    document.getElementById('layerCount').textContent = count;
}

function resize() {
    W = canvas.width = window.innerWidth;
    H = canvas.height = window.innerHeight;
    histCanvas.width = window.innerWidth - 260;
    histCanvas.height = 40;
    if (brainData) {
        fitView();
        drawHist();
        draw();
    }
}

// ── Events ──
document.getElementById('brainSelect').addEventListener('change', (e) => {
    if (e.target.value) onBrainSelected(e.target.value);
});
document.getElementById('dotSize').addEventListener('input', (e) => {
    document.getElementById('dotSizeVal').textContent = e.target.value;
    draw();
});
document.getElementById('showConns').addEventListener('change', (e) => {
    document.getElementById('connControls').style.display = e.target.checked ? 'flex' : 'none';
    draw();
});
document.getElementById('connCount').addEventListener('input', (e) => {
    document.getElementById('connCountVal').textContent = e.target.value;
    draw();
});
document.getElementById('showLabels').addEventListener('change', draw);
document.getElementById('colorMode').addEventListener('change', draw);

document.getElementById('layerSlider').addEventListener('input', (e) => {
    currentLayer = parseInt(e.target.value);
    updateLayerInfo();
    fitView();
    drawHist();
    draw();
});

// Arrow keys
document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowRight' && currentLayer < nLayers - 1) {
        currentLayer++;
    } else if (e.key === 'ArrowLeft' && currentLayer > 0) {
        currentLayer--;
    } else return;

    document.getElementById('layerSlider').value = currentLayer;
    updateLayerInfo();
    fitView();
    drawHist();
    draw();
});

// Zoom
canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const factor = e.deltaY > 0 ? 0.9 : 1.1;
    const [wx, wy] = screenToWorld(e.clientX, e.clientY);
    scale *= factor;
    offsetX = e.clientX - wx * scale;
    offsetY = e.clientY - wy * scale;
    draw();
});

// Pan
canvas.addEventListener('mousedown', (e) => {
    dragging = true;
    dragStartX = e.clientX - offsetX;
    dragStartY = e.clientY - offsetY;
});
canvas.addEventListener('mouseup', () => { dragging = false; });
canvas.addEventListener('mousemove', (e) => {
    if (dragging) {
        offsetX = e.clientX - dragStartX;
        offsetY = e.clientY - dragStartY;
        draw();
        return;
    }

    // Hover
    const neurons = getLayerNeurons();
    let closest = null;
    let closestDist = 20;

    for (let i = 0; i < neurons.length; i++) {
        const n = neurons[i];
        const [sx, sy] = worldToScreen(n.y, i * 0.5);
        const d = Math.sqrt((e.clientX - sx) ** 2 + (e.clientY - sy) ** 2);
        if (d < closestDist) {
            closestDist = d;
            closest = n;
        }
    }

    if (closest !== hoveredNeuron) {
        hoveredNeuron = closest;
        draw();
    }

    if (closest) {
        let text = `#${closest.id}`;
        const name = brainData.neuron_names[String(closest.id)];
        if (name) text += `  ${name}`;
        text += `\nPos: (${closest.x.toFixed(1)}, ${closest.y.toFixed(1)})`;
        text += `\nLayer: ${closest.layer}`;
        text += `\n${closest.type === 0 ? 'Excitatory' : 'Inhibitory'}`;
        const func = brainData.neuron_types[String(closest.id)];
        if (func) text += ` (${func})`;

        tooltip.style.display = 'block';
        tooltip.style.left = (e.clientX + 15) + 'px';
        tooltip.style.top = (e.clientY - 10) + 'px';
        tooltip.textContent = text;
    } else {
        tooltip.style.display = 'none';
    }
});

window.addEventListener('resize', resize);

// Version selector
const versionSelect = document.getElementById('versionSelect');
const brainSelect = document.getElementById('brainSelect');

loadVersions(versionSelect).then(() => {
    loadBrainList(brainSelect, versionSelect.value);
});

versionSelect.addEventListener('change', () => {
    loadBrainList(brainSelect, versionSelect.value);
});

resize();
