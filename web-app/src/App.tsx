import { lazy, Suspense } from 'react';
import { STUDIO_FLAGS } from './studio/features';

const StudioApp = lazy(() => import('./studio/StudioApp').then((module) => ({ default: module.StudioApp })));
const LegacyEditor = lazy(() => import('./LegacyEditor').then((module) => ({ default: module.LegacyEditor })));

const ROOT_FALLBACK = (
  <div className="h-screen w-screen bg-[#08090a] text-white/50 flex items-center justify-center text-xs">
    Opening…
  </div>
);

function App() {
  // Strangler shell: the legacy editor remains at /, the new studio lives at
  // /studio. VITE_STUDIO_ENABLED=false is a one-release rollback to legacy.
  // Both sides are lazy so each route only downloads its own shell.
  if (
    STUDIO_FLAGS.enabled
    && typeof window !== 'undefined'
    && window.location.pathname.startsWith('/studio')
  ) {
    return (
      <Suspense fallback={ROOT_FALLBACK}>
        <StudioApp />
      </Suspense>
    );
  }

  return (
    <Suspense fallback={ROOT_FALLBACK}>
      <LegacyEditor />
    </Suspense>
  );
}

export default App;
