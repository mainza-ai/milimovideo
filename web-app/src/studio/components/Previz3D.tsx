import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { Camera, Download, X } from 'lucide-react';
import type { Project, Shot } from '../../stores/types';

interface Props {
    shot: Shot;
    project: Project;
    onClose: () => void;
}

function toWorld(x: number, y: number): [number, number] {
    return [((x - 50) / 50) * 8, ((y - 50) / 50) * 8];
}

export default function Previz3D({ shot, project, onClose }: Props) {
    const hostRef = useRef<HTMLDivElement>(null);
    const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
    const sceneRef = useRef<THREE.Scene | null>(null);
    const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const host = hostRef.current;
        if (!host) return;

        let renderer: THREE.WebGLRenderer;
        try {
            renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
        } catch (exc) {
            setError(exc instanceof Error ? exc.message : 'WebGL unavailable');
            return;
        }
        rendererRef.current = renderer;
        renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
        renderer.setSize(host.clientWidth, host.clientHeight);
        renderer.setClearColor(0x08090a, 1);
        host.appendChild(renderer.domElement);

        const scene = new THREE.Scene();
        scene.fog = new THREE.Fog(0x08090a, 18, 42);
        const camera = new THREE.PerspectiveCamera(50, host.clientWidth / host.clientHeight, 0.1, 200);
        camera.position.set(10, 12, 14);
        camera.lookAt(0, 0, 0);
        sceneRef.current = scene;
        cameraRef.current = camera;

        const controls = new OrbitControls(camera, renderer.domElement);
        controls.target.set(0, 0.5, 0);
        controls.enableDamping = true;
        controls.dampingFactor = 0.08;

        scene.add(new THREE.HemisphereLight(0xcfe8ff, 0x08090a, 1.1));
        const key = new THREE.DirectionalLight(0xffffff, 2);
        key.position.set(6, 12, 6);
        scene.add(key);

        const floor = new THREE.Mesh(
            new THREE.PlaneGeometry(20, 20),
            new THREE.MeshStandardMaterial({ color: 0x11161a, roughness: 0.9 }),
        );
        floor.rotation.x = -Math.PI / 2;
        scene.add(floor);

        const grid = new THREE.GridHelper(20, 20, 0x06b2eb, 0x1b232a);
        grid.position.y = 0.01;
        scene.add(grid);

        const nodes = shot.blocking?.length ? shot.blocking : [
            { id: 'camera', type: 'camera' as const, label: 'Camera', x: 50, y: 88, rotation: 0 },
            { id: 'talent', type: 'character' as const, label: shot.character || 'Talent', x: 50, y: 45, rotation: 0 },
        ];

        for (const node of nodes) {
            const [x, z] = toWorld(node.x, node.y);
            if (node.type === 'character') {
                const mesh = new THREE.Mesh(
                    new THREE.CapsuleGeometry(0.32, 1.0, 6, 12),
                    new THREE.MeshStandardMaterial({ color: 0x9a7bff, roughness: 0.6 }),
                );
                mesh.position.set(x, 0.82, z);
                scene.add(mesh);
            } else if (node.type === 'camera') {
                const body = new THREE.Mesh(
                    new THREE.BoxGeometry(0.9, 0.5, 0.6),
                    new THREE.MeshStandardMaterial({ color: 0x06b2eb, roughness: 0.4 }),
                );
                body.position.set(x, 0.45, z);
                body.rotation.y = THREE.MathUtils.degToRad(node.rotation || 0);
                scene.add(body);
                const lens = new THREE.Mesh(
                    new THREE.ConeGeometry(0.22, 0.5, 12),
                    new THREE.MeshStandardMaterial({ color: 0xd8f6ff }),
                );
                lens.rotation.x = Math.PI / 2;
                lens.position.set(0, 0, 0.55);
                body.add(lens);
            } else {
                const object = new THREE.Mesh(
                    new THREE.BoxGeometry(0.8, 0.8, 0.8),
                    new THREE.MeshStandardMaterial({ color: 0xf5b84b, roughness: 0.7 }),
                );
                object.position.set(x, 0.42, z);
                scene.add(object);
            }
        }

        // Camera path across the sequence, in shot order.
        const pathPoints: THREE.Vector3[] = [];
        for (const item of project.shots) {
            const cameraNode = item.blocking?.find((node) => node.type === 'camera');
            if (!cameraNode) continue;
            const [x, z] = toWorld(cameraNode.x, cameraNode.y);
            pathPoints.push(new THREE.Vector3(x, 0.5, z));
        }
        if (pathPoints.length > 1) {
            const path = new THREE.Line(
                new THREE.BufferGeometry().setFromPoints(pathPoints),
                new THREE.LineDashedMaterial({ color: 0x06b2eb, dashSize: 0.35, gapSize: 0.2 }),
            );
            path.computeLineDistances();
            scene.add(path);
            for (const point of pathPoints) {
                const dot = new THREE.Mesh(
                    new THREE.SphereGeometry(0.12, 10, 10),
                    new THREE.MeshStandardMaterial({ color: 0x06b2eb }),
                );
                dot.position.copy(point);
                scene.add(dot);
            }
        }

        let raf = 0;
        const animate = () => {
            raf = requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        };
        animate();

        const resize = () => {
            if (!host.clientWidth || !host.clientHeight) return;
            camera.aspect = host.clientWidth / host.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(host.clientWidth, host.clientHeight);
        };
        const observer = new ResizeObserver(resize);
        observer.observe(host);

        return () => {
            cancelAnimationFrame(raf);
            observer.disconnect();
            controls.dispose();
            renderer.dispose();
            renderer.domElement.remove();
            scene.traverse((object) => {
                if (object instanceof THREE.Mesh || object instanceof THREE.Line) {
                    object.geometry?.dispose();
                    const material = object.material;
                    if (Array.isArray(material)) material.forEach((item) => item.dispose());
                    else material?.dispose();
                }
            });
            rendererRef.current = null;
            sceneRef.current = null;
            cameraRef.current = null;
        };
    }, [project.shots, shot]);

    const saveDepth = () => {
        const renderer = rendererRef.current;
        const scene = sceneRef.current;
        const camera = cameraRef.current;
        if (!renderer || !scene || !camera) return;
        const width = Math.max(320, Math.round(renderer.domElement.clientWidth || 960));
        const height = Math.max(180, Math.round(renderer.domElement.clientHeight || 540));
        const target = new THREE.WebGLRenderTarget(width, height);
        const pixels = new Uint8Array(width * height * 4);
        const previousOverride = scene.overrideMaterial;
        scene.overrideMaterial = new THREE.MeshDepthMaterial();
        renderer.setRenderTarget(target);
        renderer.render(scene, camera);
        renderer.readRenderTargetPixels(target, 0, 0, width, height, pixels);
        renderer.setRenderTarget(null);
        scene.overrideMaterial = previousOverride;
        target.dispose();

        const source = document.createElement('canvas');
        source.width = width;
        source.height = height;
        const sourceContext = source.getContext('2d');
        if (!sourceContext) return;
        const image = sourceContext.createImageData(width, height);
        for (let y = 0; y < height; y += 1) {
            const flipped = height - 1 - y;
            for (let x = 0; x < width; x += 1) {
                const from = (flipped * width + x) * 4;
                const to = (y * width + x) * 4;
                image.data[to] = pixels[from];
                image.data[to + 1] = pixels[from + 1];
                image.data[to + 2] = pixels[from + 2];
                image.data[to + 3] = 255;
            }
        }
        sourceContext.putImageData(image, 0, 0);
        const anchor = document.createElement('a');
        anchor.href = source.toDataURL('image/png');
        anchor.download = `depth-${shot.id}.png`;
        anchor.click();
    };

    return (
        <div className="st-lens-backdrop" role="dialog" aria-modal="true" aria-label="3D previz">
            <div className="st-lens" style={{ width: 'min(1100px, 100%)' }}>
                <div className="st-lens-head">
                    <div>
                        <strong>3D previz · {shot.action || shot.prompt || shot.id.slice(0, 10)}</strong>
                        <div style={{ color: 'var(--st-faint)', fontSize: 11 }}>
                            Lazy-loaded three.js stage from the blocking lens. Camera path follows shot order.
                        </div>
                    </div>
                    <div className="st-button-row">
                        <button
                            type="button"
                            className="st-button"
                            data-size="sm"
                            onClick={() => {
                                const data = rendererRef.current?.domElement.toDataURL('image/png');
                                if (!data) return;
                                const anchor = document.createElement('a');
                                anchor.href = data;
                                anchor.download = `previz-${shot.id}.png`;
                                anchor.click();
                            }}
                        >
                            <Download size={12} /> Snapshot
                        </button>
                        <button type="button" className="st-button" data-size="sm" onClick={saveDepth}>
                            <Download size={12} /> Depth PNG
                        </button>
                        <button type="button" className="st-icon-button" aria-label="Close 3D previz" onClick={onClose}>
                            <X size={14} />
                        </button>
                    </div>
                </div>
                {error ? (
                    <div className="st-empty" style={{ minHeight: 240 }}>
                        <strong>3D unavailable on this device</strong>
                        <span>{error}. The editor remains usable; previz fails closed.</span>
                    </div>
                ) : (
                    <div
                        ref={hostRef}
                        style={{ width: '100%', height: 'min(62dvh, 620px)', background: '#08090a' }}
                    />
                )}
                <div className="st-lens-body" style={{ paddingTop: 0 }}>
                    <div className="st-value" style={{ color: 'var(--st-muted)', fontSize: 11 }}>
                        <Camera size={12} /> Drag to orbit · scroll to zoom · right-drag to pan. Nodes come from the blocking lens.
                    </div>
                </div>
            </div>
        </div>
    );
}
