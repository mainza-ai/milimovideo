import { useEffect, useState } from 'react';
import { Bell, BellRing, Volume2 } from 'lucide-react';
import {
    notificationPreferences,
    osNotificationsSupported,
    requestOsPermission,
    saveNotificationPreferences,
    type NotificationPreferences,
} from '../notifications/preferences';
import { useStudioDocumentStore } from '../state/documentStore';
import { isDesktop } from '../../desktop';
import { useStudioProject } from '../data/useStudioData';

export function NotificationSettingsPanel() {
    const [prefs, setPrefs] = useState<NotificationPreferences>(() => notificationPreferences());
    const [permission, setPermission] = useState<NotificationPermission>(
        osNotificationsSupported() ? Notification.permission : 'denied',
    );
    const addToast = useStudioDocumentStore((state) => state.addToast);
    const project = useStudioProject();
    const muted = prefs.mutedProjects.includes(project.id);
    const desktop = isDesktop();
    const available = desktop || osNotificationsSupported();
    const granted = desktop || permission === 'granted';

    useEffect(() => {
        const handler = (event: Event) => {
            const detail = (event as CustomEvent).detail as NotificationPreferences | undefined;
            if (detail) setPrefs(detail);
        };
        window.addEventListener('milimo:notification-preferences', handler);
        return () => window.removeEventListener('milimo:notification-preferences', handler);
    }, []);

    const update = (patch: Partial<NotificationPreferences>) => {
        setPrefs(saveNotificationPreferences(patch));
    };

    const enableOs = async () => {
        const result = await requestOsPermission();
        setPermission(result);
        if (result === 'granted') {
            update({ os: true });
            addToast('Desktop notifications enabled', 'success');
        } else {
            update({ os: false });
            addToast('Desktop notifications were not granted', 'info');
        }
    };

    return (
        <div className="st-panel" style={{ marginBottom: 14 }} data-testid="notification-settings">
            <h3 className="st-panel-title">Notifications</h3>
            <div className="st-button-row" style={{ gap: 14, flexWrap: 'wrap' }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12 }}>
                    <input
                        type="checkbox"
                        aria-label="In-app toasts"
                        checked={prefs.toasts}
                        onChange={(event) => update({ toasts: event.target.checked })}
                    />
                    <Bell size={12} /> In-app toasts
                </label>
                <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12 }}>
                    <input
                        type="checkbox"
                        aria-label="Sound cue"
                        checked={prefs.sound}
                        onChange={(event) => update({ sound: event.target.checked })}
                    />
                    <Volume2 size={12} /> Sound cue
                </label>
                <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12 }}>
                    <input
                        type="checkbox"
                        aria-label="Desktop notifications"
                        checked={prefs.os && granted}
                        disabled={!available || (!desktop && permission === 'denied')}
                        onChange={(event) => update({ os: event.target.checked })}
                    />
                    <BellRing size={12} /> Desktop notifications
                </label>
                <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12 }}>
                    <input
                        type="checkbox"
                        aria-label="Mute this project"
                        checked={muted}
                        onChange={(event) => update({
                            mutedProjects: event.target.checked
                                ? [...new Set([...prefs.mutedProjects, project.id])]
                                : prefs.mutedProjects.filter((id) => id !== project.id),
                        })}
                    />
                    Mute this project
                </label>
                {!desktop && osNotificationsSupported() && permission !== 'granted' && (
                    <button type="button" className="st-button" data-size="sm" onClick={() => void enableOs()}>
                        Enable desktop
                    </button>
                )}
            </div>
            <p className="st-value" style={{ marginTop: 8, color: 'var(--st-faint)', fontSize: 11 }}>
                Toasts and the activity center are always available. Desktop notifications fire only
                when the tab is in the background and the browser grants permission. Sound plays on
                success and failure only.
            </p>
        </div>
    );
}
