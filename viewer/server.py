"""
viewer/server.py — Brain viewer + builder web server (V8).

Serves 3D/2D views of brain DBs and the builder UI.
Includes save/load API for sub-components and components.

Usage:
  py viewer/server.py          (from inner-models-v8/)
  py server.py                 (from viewer/)
"""
import sys, os, json, sqlite3, subprocess, signal
import http.server
from http.server import ThreadingHTTPServer
import urllib.parse

# Project root is one level up from viewer/
VIEWER_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(VIEWER_DIR)
RESEARCH_DIR = os.path.dirname(BASE)  # C:/LIFE/Research/
sys.path.insert(0, BASE)

# Version -> brain folder mappings
VERSIONS = {
    'v10': {
        'root': BASE,
        'folders': ['brains'],
    },
    'v8': {
        'root': os.path.join(RESEARCH_DIR, 'inner-models-v8'),
        'folders': ['brains'],
    },
    'v7': {
        'root': os.path.join(RESEARCH_DIR, 'inner-models-v7'),
        'folders': ['samples', 'brains'],
    },
}

# Save/load directories for builder tiers
SAVE_DIRS = {
    'sub_components': os.path.join(BASE, 'blocks', 'sub_components'),
    'components':     os.path.join(BASE, 'blocks', 'components'),
    'brains':         os.path.join(BASE, 'blocks', 'brains'),
}

MIME_TYPES = {
    '.html': 'text/html',
    '.css': 'text/css',
    '.js': 'application/javascript',
    '.json': 'application/json',
    '.png': 'image/png',
}


def list_brains(version='v8'):
    """Return list of available brains for a given version."""
    ver = VERSIONS.get(version)
    if not ver:
        return []
    root = ver['root']
    groups = []
    for folder in ver['folders']:
        full = os.path.join(root, folder)
        if not os.path.isdir(full):
            continue
        dbs = sorted([f for f in os.listdir(full)
                      if f.endswith('.db') and not f.startswith('_')
                      and f != 'results.db'])
        if dbs:
            groups.append({'folder': folder, 'brains': dbs})
    return {'version': version, 'groups': groups}


def load_brain_data(db_path):
    """Load brain from DB into JSON-serializable dict. Handles V7 and V8 schemas."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute('PRAGMA table_info(neurons)')
    cols = [row[1] for row in cur.fetchall()]
    has_pos = 'pos_x' in cols
    has_neuron_type = 'neuron_type' in cols

    cur.execute('SELECT * FROM neurons ORDER BY id')
    neurons_raw = cur.fetchall()
    col_names = [d[0] for d in cur.description]

    neurons = []
    for row in neurons_raw:
        r = dict(zip(col_names, row))
        n = {
            'id': r['id'],
            'x': r.get('pos_x', 0.0) or 0.0,
            'y': r.get('pos_y', 0.0) or 0.0,
            'z': r.get('pos_z', 0.0) or 0.0,
        }
        # V8 has neuron_type (RS/FS/IB/CH/LTS), V7 has type (0/1)
        if has_neuron_type:
            n['type'] = r['neuron_type']
        else:
            n['type'] = r.get('type', 0)
        neurons.append(n)

    # Synapse loading — check for synapse_type column (V8)
    cur.execute('PRAGMA table_info(synapses)')
    syn_cols = [row[1] for row in cur.fetchall()]
    has_synapse_type = 'synapse_type' in syn_cols

    cur.execute('''SELECT source, target, weight
                   FROM synapses ORDER BY ABS(weight) DESC LIMIT 1000''')
    connections = [{'src': s, 'tgt': t, 'w': w} for s, t, w in cur.fetchall()]

    # Count exc/inh
    if has_neuron_type:
        cur.execute("SELECT COUNT(*) FROM neurons WHERE neuron_type IN ('RS','IB','CH')")
        n_exc = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM neurons WHERE neuron_type IN ('FS','LTS')")
        n_inh = cur.fetchone()[0]
    else:
        cur.execute('SELECT COUNT(*) FROM neurons WHERE type=0')
        n_exc = cur.fetchone()[0]
        cur.execute('SELECT COUNT(*) FROM neurons WHERE type=1')
        n_inh = cur.fetchone()[0]

    cur.execute('SELECT COUNT(*) FROM synapses')
    n_syn = cur.fetchone()[0]

    # Synapse type breakdown (V8 only)
    synapse_types = {}
    if has_synapse_type:
        cur.execute('SELECT synapse_type, COUNT(*) FROM synapses GROUP BY synapse_type')
        synapse_types = {row[0]: row[1] for row in cur.fetchall()}

    conn.close()

    return {
        'neurons': neurons,
        'connections': connections,
        'n': len(neurons),
        'n_exc': n_exc,
        'n_inh': n_inh,
        'n_syn': n_syn,
        'has_pos': has_pos,
        'synapse_types': synapse_types,
    }


# ── Save/Load for builder tiers ──

def save_block(tier, name, data):
    """Save a builder block (sub-component, component, or brain layout) as JSON."""
    save_dir = SAVE_DIRS.get(tier)
    if not save_dir:
        return {'error': f'Unknown tier: {tier}'}
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'{name}.json')
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    return {'ok': True, 'path': path}


def list_blocks(tier):
    """List saved blocks for a tier."""
    save_dir = SAVE_DIRS.get(tier)
    if not save_dir or not os.path.isdir(save_dir):
        return {'items': []}
    items = []
    for f in sorted(os.listdir(save_dir)):
        if not f.endswith('.json'):
            continue
        try:
            with open(os.path.join(save_dir, f)) as fh:
                data = json.load(fh)
            items.append({
                'name': f[:-5],  # strip .json
                'nodes': len(data.get('nodes', data.get('clusters', []))),
                'paths': len(data.get('paths', data.get('connections', []))),
            })
        except Exception:
            items.append({'name': f[:-5], 'nodes': '?', 'paths': '?'})
    return {'items': items}


def load_block(tier, name):
    """Load a saved block by tier and name."""
    save_dir = SAVE_DIRS.get(tier)
    if not save_dir:
        return {'error': f'Unknown tier: {tier}'}
    path = os.path.join(save_dir, f'{name}.json')
    if not os.path.exists(path):
        return {'error': f'Not found: {name}'}
    with open(path) as f:
        data = json.load(f)
    return {'name': name, 'data': data}


# ── Simulation process management ──
PROGRAMS_DIR = os.path.join(BASE, 'programs')
SIMULATE_PY = os.path.join(BASE, 'simulate.py')
sim_process = None


def list_programs():
    """Return available programs from the programs/ folder."""
    if not os.path.isdir(PROGRAMS_DIR):
        return []
    programs = []
    for f in sorted(os.listdir(PROGRAMS_DIR)):
        if not f.endswith('.json'):
            continue
        try:
            with open(os.path.join(PROGRAMS_DIR, f)) as fh:
                prog = json.load(fh)
            programs.append({
                'file': f,
                'name': prog.get('name', f),
                'description': prog.get('description', ''),
            })
        except Exception:
            pass
    return programs


def start_simulation(brain_path, version='v8', program_file=None):
    """Spawn simulate.py as a subprocess."""
    global sim_process
    stop_simulation()

    ver = VERSIONS.get(version)
    root = ver['root'] if ver else BASE

    cmd = [sys.executable, SIMULATE_PY, brain_path]
    if program_file:
        cmd.extend(['--program', f'programs/{program_file}'])

    print(f"Starting simulation: {' '.join(cmd)}")
    sim_process = subprocess.Popen(
        cmd, cwd=root,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0,
    )
    return sim_process.pid


def stop_simulation():
    """Kill the running simulation if any."""
    global sim_process
    if sim_process is None:
        return False
    try:
        if sys.platform == 'win32':
            sim_process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            sim_process.terminate()
        sim_process.wait(timeout=3)
    except Exception:
        try:
            sim_process.kill()
        except Exception:
            pass
    sim_process = None
    return True


class BrainHandler(http.server.BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        msg = fmt % args
        if '404' in msg or '/api/' in msg:
            print(msg)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        params = urllib.parse.parse_qs(parsed.query)

        # ── Pages ──
        if path == '/' or path == '/3d':
            return self.serve_file('page_3d.html')
        if path == '/2d':
            return self.serve_file('page_2d.html')
        if path == '/build':
            return self.serve_file('builder/page.html')

        # ── API ──
        if path == '/api/versions':
            return self.serve_json(list(VERSIONS.keys()))

        if path == '/api/brains':
            version = params.get('version', ['v8'])[0]
            return self.serve_json(list_brains(version))

        if path == '/api/brain':
            brain_path = params.get('path', [''])[0]
            version = params.get('version', ['v8'])[0]
            ver = VERSIONS.get(version)
            root = ver['root'] if ver else BASE
            full_path = os.path.join(root, brain_path)
            if not os.path.exists(full_path):
                self.send_response(404)
                self.end_headers()
                return
            return self.serve_json(load_brain_data(full_path))

        if path == '/api/programs':
            return self.serve_json(list_programs())

        if path == '/api/sim_status':
            running = sim_process is not None and sim_process.poll() is None
            return self.serve_json({'running': running, 'pid': sim_process.pid if running else None})

        # ── Builder save/load API ──
        if path == '/api/build/list':
            tier = params.get('tier', ['sub_components'])[0]
            return self.serve_json(list_blocks(tier))

        if path == '/api/build/load':
            tier = params.get('tier', ['sub_components'])[0]
            name = params.get('name', [''])[0]
            return self.serve_json(load_block(tier, name))

        # ── Static files ──
        clean = path.lstrip('/')
        file_path = os.path.join(VIEWER_DIR, clean)
        if os.path.isfile(file_path):
            return self.serve_file(clean)

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(length)) if length > 0 else {}

        if path == '/api/run':
            brain = body.get('brain', '')
            version = body.get('version', 'v8')
            program = body.get('program', None)
            if not brain:
                return self.serve_json({'error': 'No brain selected'})
            pid = start_simulation(brain, version, program)
            return self.serve_json({'ok': True, 'pid': pid})

        if path == '/api/stop':
            stopped = stop_simulation()
            return self.serve_json({'ok': True, 'stopped': stopped})

        if path == '/api/build/save':
            tier = body.get('tier', 'sub_components')
            name = body.get('name', '')
            data = body.get('data', {})
            if not name:
                return self.serve_json({'error': 'No name provided'})
            result = save_block(tier, name, data)
            return self.serve_json(result)

        if path == '/api/build/generate':
            try:
                from schema import create_brain_db, add_neuron, add_synapse
                result = generate_brain_from_layout(body)
                if body.get('run'):
                    pid = start_simulation(result['path'], 'v8')
                    result['pid'] = pid
                return self.serve_json(result)
            except Exception as e:
                import traceback
                traceback.print_exc()
                return self.serve_json({'error': str(e)})

        self.send_response(404)
        self.end_headers()

    def serve_file(self, filename):
        file_path = os.path.join(VIEWER_DIR, filename)
        if not os.path.isfile(file_path):
            self.send_response(404)
            self.end_headers()
            return
        ext = os.path.splitext(filename)[1]
        mime = MIME_TYPES.get(ext, 'application/octet-stream')
        self.send_response(200)
        self.send_header('Content-Type', mime + '; charset=utf-8')
        self.end_headers()
        with open(file_path, 'rb') as f:
            self.wfile.write(f.read())

    def serve_json(self, data):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))


def generate_brain_from_layout(config):
    """Generate a V8 brain DB from a builder layout. Placeholder — will be expanded."""
    from schema import create_brain_db, add_neuron, add_synapse
    import sqlite3 as sql

    layout = config.get('data', config.get('layout', {}))
    name = config.get('name', 'brain_v8')
    nodes = layout.get('nodes', [])
    paths = layout.get('paths', [])

    if not nodes:
        raise ValueError("No nodes in layout")

    brains_dir = os.path.join(BASE, 'brains')
    os.makedirs(brains_dir, exist_ok=True)
    db_path = os.path.join(brains_dir, f'{name}.db')

    conn = create_brain_db(db_path)

    # Add neurons — builder sends 'type', server uses 'neuron_type'
    id_map = {}
    for node in nodes:
        nid = add_neuron(conn,
            neuron_type=node.get('neuron_type', node.get('type', 'RS')),
            pos_x=node.get('x', 0.0),
            pos_y=node.get('y', 0.0),
            pos_z=node.get('z', 0.0),
        )
        id_map[node.get('id', nid)] = nid

    # Add synapses — builder sends 'from'/'to'/'pathType', server uses 'source'/'target'/'synapse_type'
    for path in paths:
        src_key = path.get('source', path.get('from'))
        tgt_key = path.get('target', path.get('to'))
        src = id_map.get(src_key, src_key)
        tgt = id_map.get(tgt_key, tgt_key)
        add_synapse(conn,
            source=src,
            target=tgt,
            weight=path.get('weight', 1.0),
            delay=path.get('delay', 1),
            synapse_type=path.get('synapse_type', path.get('pathType', 'fixed')),
            params_override=path.get('params', None),
        )

    conn.commit()

    cur = conn.cursor()
    cur.execute('SELECT COUNT(*) FROM neurons')
    n_neurons = cur.fetchone()[0]
    cur.execute('SELECT COUNT(*) FROM synapses')
    n_synapses = cur.fetchone()[0]
    conn.close()

    rel_path = f'brains/{name}.db'
    return {
        'path': rel_path,
        'n_neurons': n_neurons,
        'n_synapses': n_synapses,
    }


def main():
    port = 8877
    server = ThreadingHTTPServer(('127.0.0.1', port), BrainHandler)
    print(f"Brain Viewer V8 running at http://localhost:{port}")
    print(f"  3D: http://localhost:{port}/")
    print(f"  2D: http://localhost:{port}/2d")
    print(f"  Builder: http://localhost:{port}/build")
    print("Press Ctrl+C to stop")

    import webbrowser
    webbrowser.open(f'http://localhost:{port}')

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == '__main__':
    main()
