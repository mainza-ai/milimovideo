import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { ErrorBoundary } from './components/ErrorBoundary.tsx'
import { SSEProvider } from './providers/SSEProvider.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ErrorBoundary>
      <SSEProvider>
        <App />
      </SSEProvider>
    </ErrorBoundary>
  </StrictMode>,
)
