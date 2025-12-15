'use client';

import React, { useEffect, useState } from 'react';
import { AnimatePresence, motion } from 'motion/react';
import { type ReceivedMessage } from '@livekit/components-react';
import { cn } from '@/lib/utils';

const MotionCaption = motion.create('div');

interface LiveCaptionsProps {
  messages: ReceivedMessage[];
  className?: string;
}

export function LiveCaptions({ messages, className }: LiveCaptionsProps) {
  const [visibleCaption, setVisibleCaption] = useState<{
    message: string;
    isLocal: boolean;
    id: string;
  } | null>(null);
  const [lastMessageId, setLastMessageId] = useState<string | null>(null);

  // Show caption when new message arrives, hide after 4 seconds of no updates
  useEffect(() => {
    const lastMessage = messages.at(-1);
    
    if (lastMessage && lastMessage.id !== lastMessageId) {
      setLastMessageId(lastMessage.id);
      setVisibleCaption({
        message: lastMessage.message,
        isLocal: lastMessage.from?.isLocal ?? false,
        id: lastMessage.id,
      });
    } else if (lastMessage && lastMessage.id === lastMessageId) {
      // Same message ID but content might have updated (streaming)
      setVisibleCaption({
        message: lastMessage.message,
        isLocal: lastMessage.from?.isLocal ?? false,
        id: lastMessage.id,
      });
    }
  }, [messages, lastMessageId]);

  // Auto-hide caption after 4 seconds of no changes
  useEffect(() => {
    if (!visibleCaption) return;

    const timer = setTimeout(() => {
      setVisibleCaption(null);
    }, 4000);

    return () => clearTimeout(timer);
  }, [visibleCaption?.message]); // Reset timer when message content changes

  return (
    <div className={cn('pointer-events-none', className)}>
      <AnimatePresence mode="wait">
        {visibleCaption && (
          <MotionCaption
            key={visibleCaption.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.2, ease: 'easeOut' }}
            className="flex flex-col items-center"
          >
            {/* Speaker indicator */}
            <span
              className={cn(
                'mb-1 text-xs font-medium uppercase tracking-wider',
                visibleCaption.isLocal ? 'text-cyan-400' : 'text-emerald-400'
              )}
            >
              {visibleCaption.isLocal ? 'You' : 'AI'}
            </span>
            
            {/* Caption text */}
            <div
              className={cn(
                'max-w-2xl rounded-lg bg-black/70 px-4 py-2 text-center backdrop-blur-sm',
                'text-base font-medium text-white md:text-lg'
              )}
            >
              {visibleCaption.message}
            </div>
          </MotionCaption>
        )}
      </AnimatePresence>
    </div>
  );
}

