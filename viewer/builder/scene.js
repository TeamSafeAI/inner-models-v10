/**
 * scene.js — Three.js scene setup, grid, camera, raycasting.
 * Owns the renderer and animation loop. Other modules add objects to the scene.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

let scene, camera, renderer, controls;
let gridFloor;
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

// Grid config
const GRID_SIZE = 40;       // total grid extent
const GRID_DIVISIONS = 40;  // 1 unit per division

export function init(container) {
    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111111);

    // Camera
    camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 0.1, 200);
    camera.position.set(15, 20, 25);
    camera.lookAt(0, 0, 0);

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    // Controls
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.1;
    controls.target.set(0, 0, 0);
    controls.mouseButtons = {
        LEFT: THREE.MOUSE.ROTATE,
        MIDDLE: THREE.MOUSE.DOLLY,
        RIGHT: THREE.MOUSE.PAN,
    };

    // Grid
    const grid = new THREE.GridHelper(GRID_SIZE, GRID_DIVISIONS, 0x333333, 0x222222);
    scene.add(grid);

    // Invisible plane for raycasting clicks onto the grid
    const planeGeo = new THREE.PlaneGeometry(GRID_SIZE, GRID_SIZE);
    const planeMat = new THREE.MeshBasicMaterial({ visible: false });
    gridFloor = new THREE.Mesh(planeGeo, planeMat);
    gridFloor.rotation.x = -Math.PI / 2;
    gridFloor.name = 'gridFloor';
    scene.add(gridFloor);

    // Lights (subtle, clusters are unlit spheres but good to have)
    const ambient = new THREE.AmbientLight(0xffffff, 0.4);
    scene.add(ambient);
    const dir = new THREE.DirectionalLight(0xffffff, 0.6);
    dir.position.set(10, 20, 10);
    scene.add(dir);

    // Resize
    window.addEventListener('resize', () => {
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, container.clientHeight);
    });

    // Start loop
    animate();

    return { scene, camera, renderer, controls };
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

/**
 * Raycast from mouse position against the grid floor.
 * Returns {x, z} snapped to integer grid, or null.
 */
export function raycastGrid(event, container) {
    const rect = container.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const hits = raycaster.intersectObject(gridFloor);
    if (hits.length > 0) {
        const p = hits[0].point;
        return {
            x: Math.round(p.x),
            z: Math.round(p.z),
        };
    }
    return null;
}

/**
 * Raycast against a list of meshes. Returns the first hit mesh or null.
 */
export function raycastMeshes(event, container, meshes) {
    const rect = container.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const hits = raycaster.intersectObjects(meshes);
    return hits.length > 0 ? hits[0].object : null;
}

export function getScene() { return scene; }
export function getCamera() { return camera; }
export function getControls() { return controls; }
