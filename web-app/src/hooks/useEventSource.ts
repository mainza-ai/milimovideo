import { useEffect, useRef, useState } from 'react';

type EventData = {
    job_id?: string;
    progress?: number;
    status?: string;
    message?: string;
    url?: string;
    type?: string;
};

type EventType = 'log' | 'progress' | 'complete';

export const useEventSource = (url: string) => {
    const [lastEvent, setLastEvent] = useState<{ type: EventType; data: EventData } | null>(null);
    const [isConnected, setIsConnected] = useState(false);
    const eventSourceRef = useRef<EventSource | null>(null);

    useEffect(() => {
        const es = new EventSource(url);
        eventSourceRef.current = es;

        es.onopen = () => {
            console.log('SSE Connected');
            setIsConnected(true);
        };

        es.onerror = (err) => {
            console.error('SSE Error:', err);
            setIsConnected(false);
        };

        // Listen for specific event types
        // Message event is default, but we use named events
        es.addEventListener('log', (e) => {
            try {
                const data = JSON.parse(e.data);
                setLastEvent({ type: 'log', data });
            } catch (err) { console.error(err); }
        });

        es.addEventListener('progress', (e) => {
            try {
                const data = JSON.parse(e.data);
                setLastEvent({ type: 'progress', data });
            } catch (err) { console.error(err); }
        });

        es.addEventListener('complete', (e) => {
            try {
                const data = JSON.parse(e.data);
                setLastEvent({ type: 'complete', data });
            } catch (err) { console.error(err); }
        });

        return () => {
            es.close();
            setIsConnected(false);
        };
    }, [url]);

    return { lastEvent, isConnected };
};
