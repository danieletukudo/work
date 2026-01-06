'use client';

import React, { useEffect, useState } from 'react';
import { motion } from 'motion/react';
import { useSessionContext, useSessionMessages, useLocalParticipant } from '@livekit/components-react';
import type { AppConfig } from '@/app-config';
import { LiveCaptions } from '@/components/app/live-captions';
import { PreConnectMessage } from '@/components/app/preconnect-message';
import { TileLayout } from '@/components/app/tile-layout';
import {
  AgentControlBar,
  type ControlBarControls,
} from '@/components/livekit/agent-control-bar/agent-control-bar';
import { useConnection } from '@/hooks/useConnection';
import { cn } from '@/lib/utils';

const MotionBottom = motion.create('div');

const BOTTOM_VIEW_MOTION_PROPS = {
  variants: {
    visible: {
      opacity: 1,
      translateY: '0%',
    },
    hidden: {
      opacity: 0,
      translateY: '100%',
    },
  },
  initial: 'hidden',
  animate: 'visible',
  exit: 'hidden',
  transition: {
    duration: 0.3,
    delay: 0.5,
    ease: 'easeOut',
  },
};

interface FadeProps {
  top?: boolean;
  bottom?: boolean;
  className?: string;
}

export function Fade({ top = false, bottom = false, className }: FadeProps) {
  return (
    <div
      className={cn(
        'from-background pointer-events-none h-4 bg-linear-to-b to-transparent',
        top && 'bg-linear-to-b',
        bottom && 'bg-linear-to-t',
        className
      )}
    />
  );
}

interface SessionViewProps {
  appConfig: AppConfig;
}

export const SessionView = ({
  appConfig,
  ...props
}: React.ComponentProps<'section'> & SessionViewProps) => {
  const session = useSessionContext();
  const { messages } = useSessionMessages(session);
  const { isConnectionActive, startDisconnectTransition } = useConnection();
  const { localParticipant } = useLocalParticipant();
  const [cameraAutoEnabled, setCameraAutoEnabled] = useState(false);

  const controls: ControlBarControls = {
    leave: true,
    microphone: true,
    chat: false, // Hide chat toggle - using live captions instead
    camera: false, // Hide camera toggle - auto-enabled on call start
    screenShare: false, // Hide screen share toggle
  };

  // Auto-enable camera when connection becomes active
  useEffect(() => {
    if (isConnectionActive && localParticipant && !cameraAutoEnabled) {
      // Enable camera automatically when call starts
      localParticipant.setCameraEnabled(true).then(() => {
        console.log('Camera auto-enabled');
        setCameraAutoEnabled(true);
      }).catch((err) => {
        console.error('Failed to enable camera:', err);
      });
      
      // Also enable microphone
      localParticipant.setMicrophoneEnabled(true).then(() => {
        console.log('Microphone auto-enabled');
      }).catch((err) => {
        console.error('Failed to enable microphone:', err);
      });
    }
  }, [isConnectionActive, localParticipant, cameraAutoEnabled]);

  // Reset camera auto-enabled state when disconnecting
  useEffect(() => {
    if (!isConnectionActive) {
      setCameraAutoEnabled(false);
    }
  }, [isConnectionActive]);

  return (
    <section className="bg-background relative z-10 h-full w-full overflow-hidden" {...props}>
      {/* Tile Layout - AI visualizer and camera (original centered layout) */}
      <TileLayout chatOpen={false} />

      {/* Live Captions - Centered, overlapping with voice area */}
      <div className="fixed inset-x-0 bottom-28 z-40 flex justify-center px-4 md:bottom-36">
        <LiveCaptions messages={messages} />
      </div>

      {/* Bottom Controls */}
      <MotionBottom
        {...BOTTOM_VIEW_MOTION_PROPS}
        className="fixed inset-x-3 bottom-0 z-50 md:inset-x-12"
      >
        {appConfig.isPreConnectBufferEnabled && (
          <PreConnectMessage messages={messages} className="pb-4" />
        )}
        <div className="bg-background relative mx-auto max-w-2xl pb-3 md:pb-12">
          <Fade bottom className="absolute inset-x-0 top-0 h-4 -translate-y-full" />
          <AgentControlBar
            controls={controls}
            isConnectionActive={isConnectionActive}
            onDisconnect={startDisconnectTransition}
          />
        </div>
      </MotionBottom>
    </section>
  );
};
