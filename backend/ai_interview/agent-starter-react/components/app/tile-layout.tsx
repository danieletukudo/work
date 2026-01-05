import React, { useMemo } from 'react';
import { Track } from 'livekit-client';
import { AnimatePresence, motion } from 'motion/react';
import {
  BarVisualizer,
  type TrackReference,
  VideoTrack,
  useLocalParticipant,
  useTracks,
  useVoiceAssistant,
} from '@livekit/components-react';
import { cn } from '@/lib/utils';

const MotionContainer = motion.create('div');

const ANIMATION_TRANSITION = {
  type: 'spring',
  stiffness: 675,
  damping: 75,
  mass: 1,
};

export function useLocalTrackRef(source: Track.Source) {
  const { localParticipant } = useLocalParticipant();
  const publication = localParticipant.getTrackPublication(source);
  const trackRef = useMemo<TrackReference | undefined>(
    () => (publication ? { source, participant: localParticipant, publication } : undefined),
    [source, publication, localParticipant]
  );
  return trackRef;
}

interface TileLayoutProps {
  chatOpen?: boolean;
}

export function TileLayout({ chatOpen = false }: TileLayoutProps) {
  const {
    state: agentState,
    audioTrack: agentAudioTrack,
    videoTrack: agentVideoTrack,
  } = useVoiceAssistant();
  const [screenShareTrack] = useTracks([Track.Source.ScreenShare]);
  const cameraTrack: TrackReference | undefined = useLocalTrackRef(Track.Source.Camera);

  const isCameraEnabled = cameraTrack && !cameraTrack.publication.isMuted;
  const isScreenShareEnabled = screenShareTrack && !screenShareTrack.publication.isMuted;

  const animationDelay = 0.15;
  const isAvatar = agentVideoTrack !== undefined;
  const videoWidth = agentVideoTrack?.publication.dimensions?.width ?? 0;
  const videoHeight = agentVideoTrack?.publication.dimensions?.height ?? 0;

  return (
    <>
      {/* Voice Visualizer - Centered, moved up to leave space for captions */}
      <div className="pointer-events-none fixed inset-x-0 top-8 bottom-48 z-50 md:top-12 md:bottom-56">
        <div className="relative mx-auto flex h-full max-w-2xl items-center justify-center px-4 md:px-0">
          <AnimatePresence mode="popLayout">
            {!isAvatar && (
              // Audio Agent - Voice Visualizer
              <MotionContainer
                key="agent"
                layoutId="agent"
                initial={{
                  opacity: 0,
                  scale: 0,
                }}
                animate={{
                  opacity: 1,
                  scale: 4,
                }}
                transition={{
                  ...ANIMATION_TRANSITION,
                  delay: animationDelay,
                }}
                className="bg-background aspect-square h-[90px] rounded-md border border-transparent"
              >
                <BarVisualizer
                  barCount={5}
                  state={agentState}
                  options={{ minHeight: 5 }}
                  trackRef={agentAudioTrack}
                  className="flex h-full items-center justify-center gap-1"
                >
                  <span
                    className={cn([
                      'bg-muted min-h-2.5 w-2.5 rounded-full',
                      'origin-center transition-colors duration-250 ease-linear',
                      'data-[lk-highlighted=true]:bg-foreground data-[lk-muted=true]:bg-muted',
                    ])}
                  />
                </BarVisualizer>
              </MotionContainer>
            )}

            {isAvatar && (
              // Avatar Agent
              <MotionContainer
                key="avatar"
                layoutId="avatar"
                initial={{
                  scale: 1,
                  opacity: 1,
                  maskImage:
                    'radial-gradient(circle, rgba(0, 0, 0, 1) 0, rgba(0, 0, 0, 1) 20px, transparent 20px)',
                  filter: 'blur(20px)',
                }}
                animate={{
                  maskImage:
                    'radial-gradient(circle, rgba(0, 0, 0, 1) 0, rgba(0, 0, 0, 1) 500px, transparent 500px)',
                  filter: 'blur(0px)',
                  borderRadius: 12,
                }}
                transition={{
                  ...ANIMATION_TRANSITION,
                  delay: animationDelay,
                  maskImage: {
                    duration: 1,
                  },
                  filter: {
                    duration: 1,
                  },
                }}
                className="h-auto w-full max-w-md overflow-hidden bg-black drop-shadow-xl/80"
              >
                <VideoTrack
                  width={videoWidth}
                  height={videoHeight}
                  trackRef={agentVideoTrack}
                />
              </MotionContainer>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* Camera - Fixed to top right corner */}
      <div className="pointer-events-none fixed right-4 top-4 z-50 md:right-8 md:top-8">
        <AnimatePresence>
          {((cameraTrack && isCameraEnabled) || (screenShareTrack && isScreenShareEnabled)) && (
            <MotionContainer
              key="camera"
              layout="position"
              layoutId="camera"
              initial={{
                opacity: 0,
                scale: 0,
              }}
              animate={{
                opacity: 1,
                scale: 1,
              }}
              exit={{
                opacity: 0,
                scale: 0,
              }}
              transition={{
                ...ANIMATION_TRANSITION,
                delay: animationDelay,
              }}
              className="drop-shadow-lg/20"
            >
              <VideoTrack
                trackRef={cameraTrack || screenShareTrack}
                width={(cameraTrack || screenShareTrack)?.publication.dimensions?.width ?? 0}
                height={(cameraTrack || screenShareTrack)?.publication.dimensions?.height ?? 0}
                className="bg-muted aspect-video w-[120px] rounded-lg object-cover md:w-[160px]"
              />
            </MotionContainer>
          )}
        </AnimatePresence>
      </div>
    </>
  );
}
