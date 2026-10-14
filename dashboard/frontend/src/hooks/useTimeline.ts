// src/hooks/useTimeline.ts
// Timeline playback controller.
// Uses requestAnimationFrame for smooth animation — all views advance together.

import { useEffect, useRef } from 'react';
import { useSimulationStore } from '../store/simulationStore';

const MS_PER_FRAME = 1000 / 30; // ~30 fps cap

export function useTimeline() {
  const { result, currentIndex, isPlaying, playbackSpeed, setCurrentIndex, setPlaying } =
    useSimulationStore();

  const rafRef = useRef<number | null>(null);
  const lastTimeRef = useRef<number | null>(null);

  useEffect(() => {
    if (!isPlaying || !result) {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
        lastTimeRef.current = null;
      }
      return;
    }

    const totalSteps = result.timestamps.length;

    const tick = (now: number) => {
      if (lastTimeRef.current === null) {
        lastTimeRef.current = now;
        rafRef.current = requestAnimationFrame(tick);
        return;
      }

      const dt = now - lastTimeRef.current;
      if (dt >= MS_PER_FRAME) {
        lastTimeRef.current = now;
        const { currentIndex, isPlaying } = useSimulationStore.getState();

        if (!isPlaying) return; // stopped between frames

        const nextIdx = currentIndex + playbackSpeed;
        if (nextIdx >= totalSteps - 1) {
          setCurrentIndex(totalSteps - 1);
          setPlaying(false);
          return;
        }
        setCurrentIndex(Math.floor(nextIdx));
      }

      rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);

    return () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
        lastTimeRef.current = null;
      }
    };
  }, [isPlaying, playbackSpeed, result]);

  const totalSteps = result ? result.timestamps.length : 0;

  return {
    currentIndex,
    totalSteps,
    isPlaying,
    currentTimestamp: result?.timestamps[currentIndex] ?? null,
    currentElapsed: result?.elapsed_seconds[currentIndex] ?? 0,
    progress: totalSteps > 0 ? currentIndex / (totalSteps - 1) : 0,
    seek: (idx: number) => {
      setCurrentIndex(Math.max(0, Math.min(idx, totalSteps - 1)));
    },
    play: () => setPlaying(true),
    pause: () => setPlaying(false),
    stepForward: () => {
      const idx = useSimulationStore.getState().currentIndex;
      setCurrentIndex(Math.min(idx + 1, totalSteps - 1));
    },
    stepBack: () => {
      const idx = useSimulationStore.getState().currentIndex;
      setCurrentIndex(Math.max(idx - 1, 0));
    },
    reset: () => {
      setCurrentIndex(0);
      setPlaying(false);
    },
  };
}
