import { useEffect, useRef, useState } from 'react';
import { getAssetUrl } from '../../config';

type LutSource = HTMLVideoElement | HTMLImageElement;

interface Props {
    source: LutSource | null;
    lutUrl: string;
    onError?: () => void;
}

export interface ParsedLut {
    size: number;
    data: Uint8Array;
    inputMin: [number, number, number];
    inputMax: [number, number, number];
}

/**
 * Parse a `.cube` 3D LUT into an RGB8 texture. Supports DOMAIN_MIN/MAX and
 * LUT_3D_INPUT_RANGE; 1D LUTs and oversized grids are rejected explicitly.
 */
export function parseCubeLut(text: string): ParsedLut {
    let size = 0;
    let inputMin: [number, number, number] = [0, 0, 0];
    let inputMax: [number, number, number] = [1, 1, 1];
    const values: number[] = [];
    for (const raw of text.split(/\r?\n/)) {
        const line = raw.trim();
        if (!line || line.startsWith('#')) continue;
        if (line.startsWith('LUT_1D_SIZE')) {
            throw new Error('1D LUTs are not supported; export a 3D .cube');
        }
        if (line.startsWith('LUT_3D_SIZE')) {
            size = Number(line.split(/\s+/)[1]) || 0;
            continue;
        }
        if (line.startsWith('DOMAIN_MIN')) {
            const parts = line.split(/\s+/).slice(1, 4).map(Number);
            if (parts.length === 3 && parts.every(Number.isFinite)) inputMin = parts as [number, number, number];
            continue;
        }
        if (line.startsWith('DOMAIN_MAX')) {
            const parts = line.split(/\s+/).slice(1, 4).map(Number);
            if (parts.length === 3 && parts.every(Number.isFinite)) inputMax = parts as [number, number, number];
            continue;
        }
        if (line.startsWith('LUT_3D_INPUT_RANGE')) {
            const parts = line.split(/\s+/).slice(1, 3).map(Number);
            if (parts.length === 2 && parts.every(Number.isFinite)) {
                inputMin = [parts[0], parts[0], parts[0]];
                inputMax = [parts[1], parts[1], parts[1]];
            }
            continue;
        }
        if (/^[A-Z_]/i.test(line)) continue;
        const parts = line.split(/\s+/).map(Number);
        if (parts.length < 3 || parts.some((value) => Number.isNaN(value))) continue;
        values.push(parts[0], parts[1], parts[2]);
    }
    if (!size || values.length < size * size * size * 3) {
        throw new Error('LUT is missing a valid LUT_3D_SIZE or data rows');
    }
    if (size > 129) throw new Error('LUT grid is larger than 129 points');
    const data = new Uint8Array(size * size * size * 3);
    for (let index = 0; index < data.length; index += 1) {
        data[index] = Math.max(0, Math.min(255, Math.round(values[index] * 255)));
    }
    return { size, data, inputMin, inputMax };
}

const VERTEX = `#version 300 es
in vec2 a_pos;
out vec2 v_uv;
void main() {
    v_uv = vec2(a_pos.x * 0.5 + 0.5, a_pos.y * 0.5 + 0.5);
    gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

const FRAGMENT = `#version 300 es
precision highp float;
precision highp sampler3D;
in vec2 v_uv;
out vec4 outColor;
uniform sampler2D u_source;
uniform sampler3D u_lut;
uniform float u_size;
uniform vec3 u_input_min;
uniform vec3 u_input_max;
void main() {
    vec3 color = texture(u_source, v_uv).rgb;
    vec3 normalized = clamp((color - u_input_min) / max(u_input_max - u_input_min, vec3(0.00001)), 0.0, 1.0);
    vec3 coord = normalized * ((u_size - 1.0) / u_size) + (0.5 / u_size);
    outColor = vec4(texture(u_lut, coord).rgb, 1.0);
}`;

function compile(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader | null {
    const shader = gl.createShader(type);
    if (!shader) return null;
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        gl.deleteShader(shader);
        return null;
    }
    return shader;
}

function isVideo(source: LutSource): source is HTMLVideoElement {
    return typeof HTMLVideoElement !== 'undefined' && source instanceof HTMLVideoElement;
}

function sourceReady(source: LutSource): boolean {
    if (isVideo(source)) return source.readyState >= 2 && source.videoWidth > 0;
    return source.complete && source.naturalWidth > 0;
}

/**
 * Live 3D LUT preview for the Edit stage. The canvas is appended only after the
 * first successful draw, so a blocked/failed upload can never cover the video
 * with a black rectangle. Uploads are throttled and downscaled.
 */
export function LutPreview({ source, lutUrl, onError }: Props) {
    const [ready, setReady] = useState(false);
    const [failed, setFailed] = useState(false);
    const onErrorRef = useRef(onError);
    useEffect(() => { onErrorRef.current = onError; }, [onError]);

    useEffect(() => {
        setFailed(false);
        setReady(false);
    }, [lutUrl, source]);

    useEffect(() => {
        if (!source || !lutUrl || failed) return undefined;
        let cancelled = false;
        let frame = 0;
        let cleanup: (() => void) | undefined;

        const start = async () => {
            try {
                const response = await fetch(getAssetUrl(lutUrl) || lutUrl);
                if (!response.ok) throw new Error('LUT fetch failed');
                const lut = parseCubeLut(await response.text());
                if (cancelled) return;

                const glCanvas = document.createElement('canvas');
                const gl = glCanvas.getContext('webgl2', { premultipliedAlpha: false, preserveDrawingBuffer: true });
                if (!gl) throw new Error('WebGL2 unavailable');
                // Draw the WebGL result into a visible 2D canvas. Some browsers
                // compose hardware video above a sibling WebGL canvas; the 2D
                // surface is the reliable display path.
                const canvas = document.createElement('canvas');
                canvas.className = 'st-lut-canvas';
                const output = canvas.getContext('2d');
                if (!output) throw new Error('2D canvas unavailable');

                const vertex = compile(gl, gl.VERTEX_SHADER, VERTEX);
                const fragment = compile(gl, gl.FRAGMENT_SHADER, FRAGMENT);
                const program = gl.createProgram();
                if (!vertex || !fragment || !program) throw new Error('LUT shader failed');
                gl.attachShader(program, vertex);
                gl.attachShader(program, fragment);
                gl.linkProgram(program);
                if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error('LUT program failed');

                const buffer = gl.createBuffer();
                gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
                gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
                    -1, -1, 1, -1, -1, 1,
                    -1, 1, 1, -1, 1, 1,
                ]), gl.STATIC_DRAW);
                const position = gl.getAttribLocation(program, 'a_pos');
                gl.enableVertexAttribArray(position);
                gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);

                const sourceTexture = gl.createTexture();
                gl.activeTexture(gl.TEXTURE0);
                gl.bindTexture(gl.TEXTURE_2D, sourceTexture);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

                const lutTexture = gl.createTexture();
                gl.activeTexture(gl.TEXTURE1);
                gl.bindTexture(gl.TEXTURE_3D, lutTexture);
                gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
                gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
                gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
                gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
                gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
                // RGBA8 is the broadly supported 3D texture format; RGB8 fails
                // on some WebGL2 implementations with INVALID_OPERATION.
                const rgba = new Uint8Array(lut.size ** 3 * 4);
                for (let index = 0, source = 0; index < rgba.length; index += 4, source += 3) {
                    rgba[index] = lut.data[source];
                    rgba[index + 1] = lut.data[source + 1];
                    rgba[index + 2] = lut.data[source + 2];
                    rgba[index + 3] = 255;
                }
                gl.texImage3D(
                    gl.TEXTURE_3D, 0, gl.RGBA8,
                    lut.size, lut.size, lut.size, 0,
                    gl.RGBA, gl.UNSIGNED_BYTE, rgba,
                );

                gl.useProgram(program);
                gl.uniform1i(gl.getUniformLocation(program, 'u_source'), 0);
                gl.uniform1i(gl.getUniformLocation(program, 'u_lut'), 1);
                gl.uniform1f(gl.getUniformLocation(program, 'u_size'), lut.size);
                gl.uniform3fv(gl.getUniformLocation(program, 'u_input_min'), lut.inputMin);
                gl.uniform3fv(gl.getUniformLocation(program, 'u_input_max'), lut.inputMax);
                gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
                (window as any).__lutDebug = { gl, program, lut, sourceTexture, lutTexture };

                const host = source.parentElement;
                if (!host) throw new Error('No stage host');

                const sourceCanvas = document.createElement('canvas');
                const context = sourceCanvas.getContext('2d', { willReadFrequently: false });
                if (!context) throw new Error('2D canvas unavailable');

                let drawn = false;
                let lastTime = -1;
                const draw = () => {
                    if (!sourceReady(source)) return false;
                    const width = isVideo(source) ? source.videoWidth : source.naturalWidth;
                    const height = isVideo(source) ? source.videoHeight : source.naturalHeight;
                    const scale = Math.min(1, 1280 / Math.max(1, width));
                    const targetWidth = Math.max(1, Math.round(width * scale));
                    const targetHeight = Math.max(1, Math.round(height * scale));
                    if (sourceCanvas.width !== targetWidth || sourceCanvas.height !== targetHeight) {
                        sourceCanvas.width = targetWidth;
                        sourceCanvas.height = targetHeight;
                        glCanvas.width = targetWidth;
                        glCanvas.height = targetHeight;
                        canvas.width = targetWidth;
                        canvas.height = targetHeight;
                        gl.viewport(0, 0, targetWidth, targetHeight);
                    }
                    context.drawImage(source, 0, 0, targetWidth, targetHeight);
                    gl.activeTexture(gl.TEXTURE0);
                    gl.bindTexture(gl.TEXTURE_2D, sourceTexture);
                    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, sourceCanvas);
                    gl.drawArrays(gl.TRIANGLES, 0, 6);
                    output.drawImage(glCanvas, 0, 0);
                    if (!drawn) {
                        drawn = true;
                        host.appendChild(canvas);
                        // Hardware video overlays can paint above a sibling canvas;
                        // hide the source once the filtered frame is visible.
                        source.style.visibility = 'hidden';
                        setReady(true);
                    }
                    return true;
                };

                const render = () => {
                    if (!cancelled && !document.hidden) {
                        if (isVideo(source)) {
                            // Upload when the frame advanced (play or seek) or on
                            // the first ready frame.
                            const changed = Math.abs(source.currentTime - lastTime) >= 1 / 60;
                            if (changed || !drawn) {
                                lastTime = source.currentTime;
                                draw();
                            }
                        } else if (!drawn) {
                            draw();
                        }
                    }
                    frame = window.requestAnimationFrame(render);
                };

                draw();
                frame = window.requestAnimationFrame(render);

                const redraw = () => { lastTime = -1; draw(); };
                if (isVideo(source)) {
                    source.addEventListener('seeked', redraw);
                    source.addEventListener('loadeddata', redraw);
                    source.addEventListener('playing', redraw);
                } else {
                    source.addEventListener('load', redraw);
                }

                cleanup = () => {
                    window.cancelAnimationFrame(frame);
                    source.style.visibility = '';
                    if (isVideo(source)) {
                        source.removeEventListener('seeked', redraw);
                        source.removeEventListener('loadeddata', redraw);
                        source.removeEventListener('playing', redraw);
                    } else {
                        source.removeEventListener('load', redraw);
                    }
                    canvas.remove();
                    gl.deleteTexture(sourceTexture);
                    gl.deleteTexture(lutTexture);
                    gl.deleteBuffer(buffer);
                    gl.deleteProgram(program);
                    gl.deleteShader(vertex);
                    gl.deleteShader(fragment);
                };
            } catch {
                if (!cancelled) {
                    setFailed(true);
                    onErrorRef.current?.();
                }
            }
        };

        void start();
        return () => {
            cancelled = true;
            cleanup?.();
        };
    }, [failed, lutUrl, source]);

    if (failed) return null;
    return ready ? <span className="st-lut-active" aria-hidden="true" /> : null;
}
