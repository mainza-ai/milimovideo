import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { ErrorBoundary } from './components/ErrorBoundary.tsx'
import SSEProvider from './providers/SSEProvider.tsx'
import { API_BASE_URL, API_TOKEN } from './config'

// Single auth boundary for all existing fetch call sites. EventSource and media
// URLs carry the token separately because headers are unavailable there.
if (API_TOKEN) {
    const nativeFetch = window.fetch.bind(window)
    window.fetch = (input: RequestInfo | URL, init: RequestInit = {}) => {
        const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url
        if (url.startsWith(API_BASE_URL)) {
            const headers = new Headers(init.headers || {})
            headers.set('Authorization', `Bearer ${API_TOKEN}`)
            init = { ...init, headers }
        }
        return nativeFetch(input, init)
    }
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ErrorBoundary>
      <SSEProvider>
        <App />
      </SSEProvider>
    </ErrorBoundary>
  </StrictMode>,
)
