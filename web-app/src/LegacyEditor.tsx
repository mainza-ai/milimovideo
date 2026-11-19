import { CinematicPlayer } from './components/Player/CinematicPlayer';
import { Layout } from './components/Layout';
import { useTimelineStore } from './stores/timelineStore';
import { useShallow } from 'zustand/react/shallow';

/**
 * Legacy NLE at `/`, split out of the entry chunk.
 *
 * The studio route must not pay for the legacy editor's component graph, and the
 * legacy route must not pay for the studio shell. Both are lazy at the router
 * boundary in `App.tsx`; this module is the legacy side.
 */
export function LegacyEditor() {
  const toasts = useTimelineStore(useShallow(state => state.toasts));

  return (
    <>
      <Layout>
        <CinematicPlayer />
      </Layout>

      {/* Toast Notification Container */}
      <div className="fixed bottom-4 right-4 flex flex-col gap-2 z-[100] pointer-events-none">
        {toasts.map(toast => (
          <div
            key={toast.id}
            className={`
              px-4 py-2 rounded-lg shadow-xl text-xs font-semibold backdrop-blur-md animate-in slide-in-from-right-10 fade-in
              ${toast.type === 'error' ? 'bg-red-500/90 text-white' :
                toast.type === 'success' ? 'bg-green-500/90 text-white' :
                  'bg-white/10 text-white border border-white/10'}
            `}
          >
            {toast.message}
          </div>
        ))}
      </div>
    </>
  );
}
