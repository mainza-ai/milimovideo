import { createContext, useContext } from 'react';
import type { SSEContextType } from '../contexts/SSEContext';

export const SSEContext = createContext<SSEContextType>({ isConnected: false, lastEventTime: 0 });

export const useSSE = () => useContext(SSEContext);
