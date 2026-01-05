'use client';

import { createContext, useContext, useMemo, useState } from 'react';
import { TokenSource } from 'livekit-client';
import { SessionProvider, useSession } from '@livekit/components-react';
import type { AppConfig } from '@/app-config';

interface ConnectionContextType {
  isConnectionActive: boolean;
  connect: (startSession?: boolean) => void;
  startDisconnectTransition: () => void;
  onDisconnectTransitionComplete: () => void;
}

const ConnectionContext = createContext<ConnectionContextType>({
  isConnectionActive: false,
  connect: () => {},
  startDisconnectTransition: () => {},
  onDisconnectTransitionComplete: () => {},
});

export function useConnection() {
  const ctx = useContext(ConnectionContext);
  if (!ctx) {
    throw new Error('useConnection must be used within a ConnectionProvider');
  }
  return ctx;
}

interface ConnectionProviderProps {
  appConfig: AppConfig;
  children: React.ReactNode;
  jobId?: string | null;
  appId?: string | null;
}

export function ConnectionProvider({ appConfig, children, jobId, appId }: ConnectionProviderProps) {
  const [isConnectionActive, setIsConnectionActive] = useState(false);

  const tokenSource = useMemo(() => {
    // Use Python Flask API server for connection details
    // This uses the Python environment which has all LiveKit credentials configured
    const PYTHON_API_ENDPOINT = process.env.NEXT_PUBLIC_CONN_DETAILS_ENDPOINT || 'http://localhost:5001/api/connection-details';
    
    return TokenSource.custom(async () => {
      try {
        const res = await fetch(PYTHON_API_ENDPOINT, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-Sandbox-Id': appConfig.sandboxId ?? '',
          },
          body: JSON.stringify({
            room_config: appConfig.agentName
              ? {
                  agents: [{ agent_name: appConfig.agentName }],
                }
              : undefined,
            jobid: jobId,
            appid: appId,
          }),
        });
        
        if (!res.ok) {
          const errorData = await res.json().catch(() => ({ error: 'Unknown error' }));
          throw new Error(errorData.error || `HTTP ${res.status}`);
        }
        
        return await res.json();
      } catch (error) {
        console.error('Error fetching connection details from Python API:', error);
        throw new Error('Error fetching connection details! Make sure the Python API server is running (python api_server.py)');
      }
    });
  }, [appConfig, jobId, appId]);

  const session = useSession(
    tokenSource,
    appConfig.agentName ? { agentName: appConfig.agentName } : undefined
  );

  const { start: startSession, end: endSession } = session;

  const value = useMemo(() => {
    return {
      isConnectionActive,
      connect: () => {
        setIsConnectionActive(true);
        startSession();
      },
      startDisconnectTransition: () => {
        setIsConnectionActive(false);
      },
      onDisconnectTransitionComplete: () => {
        endSession();
      },
    };
  }, [startSession, endSession, isConnectionActive]);

  return (
    <SessionProvider session={session}>
      <ConnectionContext.Provider value={value}>{children}</ConnectionContext.Provider>
    </SessionProvider>
  );
}
