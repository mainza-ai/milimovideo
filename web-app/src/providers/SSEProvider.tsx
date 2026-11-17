import React, { useEffect, useRef, useState } from 'react';
import { eventUrl } from '../config';
import { useTimelineStore } from '../stores/timelineStore';
import { useShallow } from 'zustand/react/shallow';
import { pollJobStatus } from '../utils/jobPoller';
import { SSEContext } from '../hooks/useSSE';
import {
    handleStudioJobEvent,
    handleStudioSystemEvent,
    type StudioJobEvent,
} from '../studio/events/jobBridge';

const SSEProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const [isConnected, setIsConnected] = useState(false);
    const [lastEventTime, setLastEventTime] = useState(0);
    const eventSourceRef = useRef<EventSource | null>(null);
    const lastEventIdRef = useRef<string>('');
    const handleServerEvent = useTimelineStore(useShallow((state) => state.handleServerEvent));
    const handleModelEvent = useTimelineStore(useShallow((state) => state.handleModelEvent));

    // Sync active jobs on mount (legacy store).
    useEffect(() => {
        const syncActiveShots = async () => {
            const { project } = useTimelineStore.getState();
            const activeShots = project.shots.filter((s) => s.isGenerating && s.lastJobId);
            for (const shot of activeShots) {
                if (shot.lastJobId) pollJobStatus(shot.lastJobId, shot.id);
            }
        };
        void syncActiveShots();
    }, []);

    useEffect(() => {
        let retryTimeout: ReturnType<typeof setTimeout> | undefined;
        let retryCount = 0;

        const mapLegacy = (event: StudioJobEvent) => {
            if (event.event === 'job.completed') {
                handleServerEvent('complete', {
                    job_id: event.job_id,
                    url: event.url,
                    thumbnail_url: event.thumbnail_url,
                    type: event.type || 'video',
                    shot_id: event.shot_id,
                    actual_frames: event.actual_frames,
                });
            } else if (event.event === 'job.failed') {
                handleServerEvent('error', {
                    job_id: event.job_id,
                    message: event.error || event.message || 'Generation failed',
                });
            } else if (event.event === 'job.cancelled') {
                handleServerEvent('cancelled', { job_id: event.job_id });
            } else {
                handleServerEvent('progress', {
                    job_id: event.job_id,
                    progress: event.progress ?? 0,
                    status: 'processing',
                    message: event.message,
                    enhanced_prompt: event.enhanced_prompt,
                });
            }
        };

        const connect = () => {
            if (eventSourceRef.current?.readyState === EventSource.OPEN) return;
            const query = lastEventIdRef.current
                ? `?last_event_id=${encodeURIComponent(lastEventIdRef.current)}`
                : '';
            console.log('[SSE] Connecting to /events...', lastEventIdRef.current || '');
            const es = new EventSource(eventUrl(`/events${query}`));

            const mark = (event: MessageEvent) => {
                if (event.lastEventId) lastEventIdRef.current = event.lastEventId;
                setLastEventTime(Date.now());
            };

            es.onopen = () => {
                console.log('[SSE] Connected!');
                setIsConnected(true);
                retryCount = 0;
                if (typeof window !== 'undefined') {
                    window.dispatchEvent(new CustomEvent('milimo:resync'));
                }
                const { project } = useTimelineStore.getState();
                for (const shot of project.shots) {
                    if (shot.isGenerating && shot.lastJobId) {
                        pollJobStatus(shot.lastJobId, shot.id);
                    }
                }
            };

            es.onerror = (err) => {
                console.warn('[SSE] Connection Error:', err);
                setIsConnected(false);
                es.close();
                eventSourceRef.current = null;
                const delay = Math.min(1000 * Math.pow(2, retryCount), 10000);
                retryCount += 1;
                retryTimeout = setTimeout(connect, delay);
            };

            const listen = (type: string, handler: (data: Record<string, unknown>, raw: MessageEvent) => void) => {
                es.addEventListener(type, (event: MessageEvent) => {
                    mark(event);
                    // A native EventSource `error` event has no data; onerror
                    // owns reconnection. Only parse real named-event payloads.
                    if (typeof event.data !== 'string' || event.data === 'undefined' || event.data === '') {
                        return;
                    }
                    try {
                        handler(JSON.parse(event.data), event);
                    } catch (err) {
                        console.error(`[SSE] Failed to parse ${type}`, err);
                    }
                });
            };

            const jobTypes = [
                'job.queued', 'job.waiting', 'job.started', 'job.stage', 'job.progress',
                'job.retrying', 'job.cancelling', 'job.completed', 'job.failed', 'job.cancelled',
            ];
            for (const type of jobTypes) {
                listen(type, (data) => {
                    const event = { ...(data as unknown as StudioJobEvent), event: type };
                    handleStudioJobEvent(event);
                    mapLegacy(event);
                });
            }

            listen('progress', (data) => handleServerEvent('progress', data));
            listen('complete', (data) => handleServerEvent('complete', data));
            listen('error', (data) => handleServerEvent('error', data));
            listen('cancelled', (data) => handleServerEvent('cancelled', data));
            listen('log', () => undefined);

            // Model/download/render/memory/system events: Studio refresh + legacy.
            for (const type of [
                'download_queued', 'download_progress', 'download_complete', 'download_error',
                'queue_updated', 'model_status_change', 'lora_updated',
                'render_progress', 'render_completed', 'render_failed', 'render_cancelled',
                'ai_parse_complete', 'ai_parse_failed', 'depth_complete', 'depth_failed',
                'memory.pressure', 'memory.evicted', 'queue.paused', 'queue.resumed',
            ]) {
                listen(type, (data) => {
                    handleStudioSystemEvent(type, data);
                    if (['download_progress', 'download_complete', 'download_error', 'model_status_change'].includes(type)) {
                        handleModelEvent(type, data);
                    }
                });
            }

            eventSourceRef.current = es;
        };

        connect();

        return () => {
            console.log('[SSE] Cleaning up...');
            if (eventSourceRef.current) {
                eventSourceRef.current.close();
                eventSourceRef.current = null;
            }
            if (retryTimeout) clearTimeout(retryTimeout);
        };
    }, [handleModelEvent, handleServerEvent]);

    return (
        <SSEContext.Provider value={{ isConnected, lastEventTime }}>
            {children}
        </SSEContext.Provider>
    );
};

export default SSEProvider;
