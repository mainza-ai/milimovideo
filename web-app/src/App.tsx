import { lazy, Suspense } from 'react';
import { CinematicPlayer } from './components/Player/CinematicPlayer';
import { Layout } from './components/Layout';
import { useTimelineStore } from './stores/timelineStore';

import { useShallow } from 'zustand/react/shallow';

const StudioApp = lazy(() => import('./studio/StudioApp').then((module) => ({ default: module.StudioApp })));

function App() {
  const toasts = useTimelineStore(useShallow(state => state.toasts));

  // Strangler shell: the legacy editor remains at /, the new studio lives at
  // /studio until the F1-F4 surfaces reach parity.
  if (typeof window !== 'undefined' && window.location.pathname.startsWith('/studio')) {
    return (
      <Suspense fallback={<div className="h-screen w-screen bg-[#08090a] text-white/50 flex items-center justify-center text-xs">Opening studio…</div>}>
        <StudioApp />
      </Suspense>
    );
  }

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

export default App;
