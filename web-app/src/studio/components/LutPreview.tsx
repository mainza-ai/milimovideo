import { useEffect, useState } from 'react';
import { getAssetUrl } from '../../config';

interface Props {
    video: HTMLVideoElement | null;
    lutUrl: string;
    onError?: () => void;
}

interface ParsedLut {
    size: number;
    data: Uint8Array;
}

/**
 * Parse a `.cube` 3D LUT into an RGB8 texture. Other LUT formats stay
 * export-only; the caller shows a visible "preview unavailable" badge.
 */
export function parseCubeLut(text: string): ParsedLut {
    let size = 0;
    const values: number[] = [];
    for (const raw of text.split(/\r?\n/)) {
        const line = raw.trim();
        if (!line || line.startsWith('#')) continue;
        if (line.startsWith('LUT_3D_SIZE')) {
            size = Number(line.split(/\s+/)[1]) || 0;
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
    const data = new Uint8Array(size * size * size * 3);
    for (let index = 0; index < data.length; index += 1) {
        data[index] = Math.max(0, Math.min(255, Math.round(values[index] * 255)));
    }
    return { size, data };
}

const VERTEX = `#version 300 es
in vec2 a_pos;
out vec2 v_uv;
void main() {
    v_uv = vec2(a_pos.x * 0.5 + 0.5, 0.5 - a_pos.y * 0.5);
    gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

const FRAGMENT = `#version 300 es
precision highp float;
precision highp sampler3D;
in vec2 v_uv;
out vec4 outColor;
uniform sampler2D u_video;
uniform sampler3D u_lut;
uniform float u_size;
void main() {
    vec3 color = texture(u_video, v_uv).rgb;
    vec3 coord = color * ((u_size - 1.0) / u_size) + (0.5 / u_size);
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

/**
 * Live 3D LUT preview for the Edit stage. The source video keeps playing and
 * the canvas samples it each animation frame. If WebGL or the LUT is
 * unavailable the caller keeps the raw video and shows an export-only badge.
 */
export function LutPreview({ video, lutUrl, onError }: Props) {
    const [ready, setReady] = useState(false);
    const [failed, setFailed] = useState(false);

    useEffect(() => {
        if (!video || !lutUrl || failed) return undefined;
        let cancelled = false;
        let frame = 0;
        let cleanup: (() => void) | undefined;

        const start = async () => {
            try {
                const response = await fetch(getAssetUrl(lutUrl) || lutUrl);
                if (!response.ok) throw new Error('LUT fetch failed');
                const lut = parseCubeLut(await response.text());
                if (cancelled) return;

                const canvas = document.createElement('canvas');
                canvas.className = 'st-lut-canvas';
                const gl = canvas.getContext('webgl2', { premultipliedAlpha: false });
                if (!gl) throw new Error('WebGL2 unavailable');

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

                const videoTexture = gl.createTexture();
                gl.activeTexture(gl.TEXTURE0);
                gl.bindTexture(gl.TEXTURE_2D, videoTexture);
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
                gl.texImage3D(
                    gl.TEXTURE_3D, 0, gl.RGB8,
                    lut.size, lut.size, lut.size, 0,
                    gl.RGB, gl.UNSIGNED_BYTE, lut.data,
                );

                gl.useProgram(program);
                gl.uniform1i(gl.getUniformLocation(program, 'u_video'), 0);
                gl.uniform1i(gl.getUniformLocation(program, 'u_lut'), 1);
                gl.uniform1f(gl.getUniformLocation(program, 'u_size'), lut.size);
                gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
                gl.viewport(0, 0, canvas.width, canvas.height);

                const host = video.parentElement;
                if (!host) throw new Error('No stage host');
                host.appendChild(canvas);
                setReady(true);

                const render = () => {
                    if (video.videoWidth && video.readyState >= 2) {
                        if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
                            canvas.width = video.videoWidth;
                            canvas.height = video.videoHeight;
                            gl.viewport(0, 0, canvas.width, canvas.height);
                        }
                        gl.activeTexture(gl.TEXTURE0);
                        gl.bindTexture(gl.TEXTURE_2D, videoTexture);
                        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video);
                        gl.drawArrays(gl.TRIANGLES, 0, 6);
                    }
                    frame = window.requestAnimationFrame(render);
                };
                frame = window.requestAnimationFrame(render);

                cleanup = () => {
                    window.cancelAnimationFrame(frame);
                    canvas.remove();
                    gl.deleteTexture(videoTexture);
                    gl.deleteTexture(lutTexture);
                    gl.deleteBuffer(buffer);
                    gl.deleteProgram(program);
                    gl.deleteShader(vertex);
                    gl.deleteShader(fragment);
                };
            } catch {
                if (!cancelled) {
                    setFailed(true);
                    onError?.();
                }
            }
        };

        void start();
        return () => {
            cancelled = true;
            cleanup?.();
        };
    }, [failed, lutUrl, onError, video]);

    if (failed) return null;
    return ready ? <span className="st-lut-active" aria-hidden="true" /> : null;
}
