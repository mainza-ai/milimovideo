import { Component, type ErrorInfo, type ReactNode } from 'react';

interface Props {
    children?: ReactNode;
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
