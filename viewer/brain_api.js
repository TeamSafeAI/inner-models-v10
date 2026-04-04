/**
 * brain_api.js — Shared brain loading logic for both views.
 *
 * Exports:
 *   loadVersions(selectEl)          — populate version dropdown
 *   loadBrainList(selectEl, version) — populate brain dropdown
 *   loadBrainData(path)             — fetch brain JSON from server
 *   assignLayers(brainData)         — slice neurons into layers along longest axis
 *   updateStats(brainData)          — update stats panel
 *   neuronColorRGB / neuronColorCSS — color helpers
 *   extent(arr)                     — safe {min, max, range} for large arrays
 */

let currentVersion = 'v7';

// ── Utilities ──

/** Safe min/max/range for arrays of any size (avoids stack overflow from spread). */
export function extent(arr) {
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < arr.length; i++) {
        if (arr[i] < min) min = arr[i];
        if (arr[i] > max) max = arr[i];
    }
    return { min, max, range: max - min || 1 };
}

// ── Data loading ──

export async function loadVersions(selectEl) {
    const resp = await fetch('/api/versions');
    const versions = await resp.json();
    selectEl.innerHTML = '';
    for (const v of versions) {
        const opt = document.createElement('option');
        opt.value = v;
        opt.textContent = v;
        if (v === 'v7') opt.selected = true;
        selectEl.appendChild(opt);
    }
}

export async function loadBrainList(selectEl, version) {
    currentVersion = version || 'v7';
    const resp = await fetch('/api/brains?version=' + currentVersion);
    const data = await resp.json();

    selectEl.innerHTML = '<option value="">— select brain —</option>';

    for (const g of data.groups) {
        const optgroup = document.createElement('optgroup');
        optgroup.label = g.folder;
        for (const b of g.brains) {
            const opt = document.createElement('option');
            opt.value = g.folder + '/' + b;
            opt.textContent = b.replace('.db', '');
            optgroup.appendChild(opt);
        }
        selectEl.appendChild(optgroup);
    }
}

export async function loadBrainData(path) {
    const resp = await fetch('/api/brain?version=' + currentVersion + '&path=' + encodeURIComponent(path));
    return await resp.json();
}

// ── Layer assignment ──

export function assignLayers(brainData) {
    const neurons = brainData.neurons;
    const n = neurons.length;

    const ex = extent(neurons.map(n => n.x));
    const ey = extent(neurons.map(n => n.y));
    const ez = extent(neurons.map(n => n.z));

    // Slice along longest axis
    let sliceAxis, sliceMin, sliceRange;
    if (ex.range >= ey.range && ex.range >= ez.range) {
        sliceAxis = 'x'; sliceMin = ex.min; sliceRange = ex.range;
    } else if (ey.range >= ex.range && ey.range >= ez.range) {
        sliceAxis = 'y'; sliceMin = ey.min; sliceRange = ey.range;
    } else {
        sliceAxis = 'z'; sliceMin = ez.min; sliceRange = ez.range;
    }

    const nLayers = Math.max(3, Math.min(30, Math.floor(n / 12)));

    for (const neuron of neurons) {
        const val = neuron[sliceAxis];
        neuron.layer = Math.min(
            Math.floor(((val - sliceMin) / sliceRange) * nLayers),
            nLayers - 1
        );
    }

    return { sliceAxis, nLayers, extents: { x: ex, y: ey, z: ez } };
}

export function updateStats(brainData) {
    const el = document.getElementById('brainStats');
    if (!el) return;
    el.innerHTML = `
        <div class="stat">Neurons: <span>${brainData.n}</span> (${brainData.n_exc}E / ${brainData.n_inh}I)</div>
        <div class="stat">Synapses: <span>${brainData.n_syn.toLocaleString()}</span></div>
        <div class="stat">Spatial: <span>${brainData.has_pos ? 'Yes' : 'No'}</span></div>
    `;
}

// Color helpers
const COLORS = {
    exc:     [0.29, 0.56, 0.85],
    inh:     [0.85, 0.29, 0.29],
    sensory: [0.30, 0.69, 0.31],
    motor:   [1.00, 0.60, 0.00],
    command: [0.61, 0.15, 0.69],
    inter:   [0.50, 0.50, 0.50],
};

export function neuronColorRGB(neuron, brainData, mode, nLayers) {
    let r, g, b;

    if (mode === 'ei') {
        [r, g, b] = neuron.type === 0 ? COLORS.exc : COLORS.inh;
    } else if (mode === 'func') {
        const func = brainData.neuron_types[String(neuron.id)];
        [r, g, b] = COLORS[func] || COLORS.inter;
    } else if (mode === 'layer') {
        const t = neuron.layer / Math.max(nLayers - 1, 1);
        if (t < 0.25)      { r = 0.2; g = 0.3 + t * 3;          b = 0.9; }
        else if (t < 0.5)   { r = 0.2; g = 0.9;                   b = 0.9 - (t - 0.25) * 3.6; }
        else if (t < 0.75)  { r = 0.2 + (t - 0.5) * 3.2; g = 0.9; b = 0.0; }
        else                { r = 1.0; g = 0.9 - (t - 0.75) * 3.6; b = 0.0; }
    } else {
        [r, g, b] = COLORS.exc;
    }

    return [r, g, b];
}

export function neuronColorCSS(neuron, brainData, mode, nLayers) {
    const [r, g, b] = neuronColorRGB(neuron, brainData, mode, nLayers);
    return `rgb(${Math.floor(r*255)},${Math.floor(g*255)},${Math.floor(b*255)})`;
}
