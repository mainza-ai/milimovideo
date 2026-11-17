import { isDesktop, notifyDesktop } from '../../desktop';

export interface NotificationPreferences {
    toasts: boolean;
    os: boolean;
    sound: boolean;
    mutedProjects: string[];
}

const KEY = 'milimo-studio-notifications';
const DEFAULTS: NotificationPreferences = { toasts: true, os: false, sound: false, mutedProjects: [] };

let cached: NotificationPreferences | null = null;

export function notificationPreferences(): NotificationPreferences {
    if (cached) return cached;
    if (typeof localStorage === 'undefined') return { ...DEFAULTS };
    try {
        const parsed = JSON.parse(localStorage.getItem(KEY) || '{}') as Partial<NotificationPreferences>;
        cached = {
            toasts: parsed.toasts !== false,
            os: parsed.os === true,
            sound: parsed.sound === true,
            mutedProjects: Array.isArray(parsed.mutedProjects)
                ? parsed.mutedProjects.filter((id): id is string => typeof id === 'string')
                : [],
        };
    } catch {
        cached = { ...DEFAULTS };
    }
    return cached;
}

export function saveNotificationPreferences(patch: Partial<NotificationPreferences>): NotificationPreferences {
    const next = { ...notificationPreferences(), ...patch };
    cached = next;
    if (typeof localStorage !== 'undefined') {
        localStorage.setItem(KEY, JSON.stringify(next));
    }
    if (typeof window !== 'undefined') {
        window.dispatchEvent(new CustomEvent('milimo:notification-preferences', { detail: next }));
    }
    return next;
}

export function isProjectMuted(projectId?: string | null): boolean {
    return Boolean(projectId && notificationPreferences().mutedProjects.includes(projectId));
}

export function osNotificationsSupported(): boolean {
    return typeof window !== 'undefined' && 'Notification' in window;
}

export async function requestOsPermission(): Promise<NotificationPermission> {
    if (!osNotificationsSupported()) return 'denied';
    if (Notification.permission === 'granted') return 'granted';
    try {
        return await Notification.requestPermission();
    } catch {
        return 'denied';
    }
}

/** One terminal notification, only when the tab is hidden. */
export function notifyOs(title: string, body: string, tag: string): void {
    const prefs = notificationPreferences();
    if (!prefs.os) return;
    if (typeof document === 'undefined' || !document.hidden) return;
    if (isDesktop()) {
        void notifyDesktop(title, body);
        return;
    }
    if (!osNotificationsSupported() || Notification.permission !== 'granted') return;
    try {
        const notification = new Notification(title, { body, tag });
        notification.onclick = () => {
            window.focus();
            notification.close();
        };
    } catch {
        // Notifications are best-effort; the in-app surface is authoritative.
    }
}

let audio: AudioContext | null = null;

/** A short cue; created lazily and only after a user gesture in practice. */
export function playCue(tone: 'success' | 'error'): void {
    if (!notificationPreferences().sound) return;
    try {
        const Ctx = window.AudioContext || (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
        if (!Ctx) return;
        audio = audio || new Ctx();
        const oscillator = audio.createOscillator();
        const gain = audio.createGain();
        oscillator.type = 'sine';
        oscillator.frequency.value = tone === 'success' ? 660 : 220;
        gain.gain.setValueAtTime(0.05, audio.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.0001, audio.currentTime + 0.18);
        oscillator.connect(gain).connect(audio.destination);
        oscillator.start();
        oscillator.stop(audio.currentTime + 0.18);
    } catch {
        // Audio is optional.
    }
}
