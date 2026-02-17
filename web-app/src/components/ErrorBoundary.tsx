import { Component, type ErrorInfo, type ReactNode } from 'react';

interface Props {
    children?: ReactNode;
    fallbackTitle?: string;
}

interface State {
    hasError: boolean;
    error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
    public state: State = {
        hasError: false,
        error: null
    };

    public static getDerivedStateFromError(error: Error): State {
        return { hasError: true, error };
    }

    public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
        console.error("Uncaught error:", error, errorInfo);
    }

    public render() {
        if (this.state.hasError) {
            return (
                <div className="w-screen h-screen bg-[#050505] flex items-center justify-center text-white p-8">
                    <div className="max-w-md bg-red-900/20 border border-red-500/50 rounded-lg p-6">
                        <h1 className="text-xl font-bold mb-4 text-red-500">Something went wrong</h1>
                        <pre className="text-xs bg-black/50 p-4 rounded overflow-auto mb-4 font-mono text-white/70">
                            {this.state.error?.message}
                        </pre>
                        <button
                            className="bg-white/10 hover:bg-white/20 px-4 py-2 rounded text-sm transition-colors"
                            onClick={() => window.location.reload()}
                        >
                            Reload Application
                        </button>
                    </div>
                </div>
            );
        }

        return this.props.children;
    }
}


/**
 * Granular error boundary for individual UI panels.
 * Instead of crashing the whole app, only the broken panel shows an error
 * with a "Retry" button that resets just that panel.
 */
export class PanelErrorBoundary extends Component<Props, State> {
    public state: State = {
        hasError: false,
        error: null
    };

    public static getDerivedStateFromError(error: Error): State {
        return { hasError: true, error };
    }

    public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
        console.error(`[PanelError] ${this.props.fallbackTitle || 'Panel'}:`, error, errorInfo);
    }

    public render() {
        if (this.state.hasError) {
            return (
                <div className="flex flex-col items-center justify-center h-full p-4 text-center">
                    <div className="bg-red-900/10 border border-red-500/30 rounded-lg p-4 max-w-sm">
                        <p className="text-red-400 text-sm font-medium mb-2">
                            {this.props.fallbackTitle || 'Panel'} encountered an error
                        </p>
                        <pre className="text-[10px] text-white/40 bg-black/30 p-2 rounded mb-3 overflow-auto max-h-20 font-mono">
                            {this.state.error?.message}
                        </pre>
                        <button
                            className="bg-white/10 hover:bg-white/20 px-3 py-1.5 rounded text-xs transition-colors text-white/70"
                            onClick={() => this.setState({ hasError: false, error: null })}
                        >
                            Retry
                        </button>
                    </div>
                </div>
            );
        }

        return this.props.children;
    }
}
