import React, { createContext, useContext, useEffect, useRef, useState } from 'react';
import { useTimelineStore } from '../stores/timelineStore';
import { useShallow } from 'zustand/react/shallow';
import { pollJobStatus } from '../utils/jobPoller';

interface SSEContextType {
    isConnected: boolean;
    lastEventTime: number;
}

const SSEContext = createContext<SSEContextType>({ isConnected: false, lastEventTime: 0 });

export const useSSE = () => useContext(SSEContext);

export const SSEProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const [isConnected, setIsConnected] = useState(false);
    const [lastEventTime, setLastEventTime] = useState(0);
    const eventSourceRef = useRef<EventSource | null>(null);
    const handleServerEvent = useTimelineStore(useShallow(state => state.handleServerEvent));

    // Sync active jobs on mount
    useEffect(() => {
        const syncActiveShots = async () => {
            // Check any shot that thinks it is generating
            const { project } = useTimelineStore.getState();
            const activeShots = project.shots.filter(s => s.isGenerating && s.lastJobId);
            if (activeShots.length > 0) {
                console.log(`[SSE] Syncing ${activeShots.length} active shots...`);
                for (const shot of activeShots) {
                    if (shot.lastJobId) {
                        pollJobStatus(shot.lastJobId, shot.id);
                    }
                }
            }
        };
        syncActiveShots();
    }, []); // Run once on mount

    // Connection Management
    useEffect(() => {
        let retryTimeout: any;
        let retryCount = 0;

        const connect = () => {
            if (eventSourceRef.current?.readyState === EventSource.OPEN) return;

            console.log("[SSE] Connecting to /events...");
            const es = new EventSource('http://localhost:8000/events');

            es.onopen = () => {
                console.log("[SSE] Connected!");
                setIsConnected(true);
                retryCount = 0;
            };

            es.onerror = (err) => {
                console.warn("[SSE] Connection Error:", err);
                setIsConnected(false);
                es.close();
                eventSourceRef.current = null;

                // Exponential backoff
                const delay = Math.min(1000 * Math.pow(2, retryCount), 10000);
                retryCount++;
                console.log(`[SSE] Reconnecting in ${delay}ms...`);
                retryTimeout = setTimeout(connect, delay);
            };

            // Generic Message Handler (if event type is 'message')
            es.onmessage = () => {
                // Heartbeats or generic messages
                // console.log("[SSE] Message:", event.data);
            };

            // Custom Event Handlers

            es.addEventListener('progress', (e: MessageEvent) => {
                try {
                    const data = JSON.parse(e.data);
                    handleServerEvent('progress', data);
                    setLastEventTime(Date.now());
                } catch (err) {
                    console.error("[SSE] Failed to parse progress", err);
                }
            });

            es.addEventListener('complete', (e: MessageEvent) => {
                try {
                    const data = JSON.parse(e.data);
                    handleServerEvent('complete', data);
                    setLastEventTime(Date.now());
                } catch (err) {
                    console.error("[SSE] Failed to parse complete", err);
                }
            });

            es.addEventListener('error', (e: MessageEvent) => {
                // This 'error' event type is custom from backend, NOT connection error
                try {
                    const data = JSON.parse(e.data);
                    handleServerEvent('error', data);
                } catch (err) {
                    console.error("[SSE] Failed to parse backend error", err);
                }
            });

            es.addEventListener('log', (e: MessageEvent) => {
                try {
                    JSON.parse(e.data);
                    // Optional: expose logs to UI?
                    // console.debug("[Server Log]", data.message);
                } catch (err) {
                    console.error("[SSE] Failed to parse log", err);
                }
            });

            eventSourceRef.current = es;
        };

        connect();

        return () => {
            console.log("[SSE] Cleaning up...");
            if (eventSourceRef.current) {
                eventSourceRef.current.close();
                eventSourceRef.current = null;
            }
            clearTimeout(retryTimeout);
        };
    }, []);

    return (
        <SSEContext.Provider value={{ isConnected, lastEventTime }}>
            {children}
        </SSEContext.Provider>
    );
};
