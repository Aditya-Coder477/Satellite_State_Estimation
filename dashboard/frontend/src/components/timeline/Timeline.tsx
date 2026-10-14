// src/components/timeline/Timeline.tsx
// Shared timeline controller.
// When the slider changes, ALL four views update simultaneously via Zustand.

import React from 'react';
import { useSimulationStore } from '../../store/simulationStore';
import { useTimeline } from '../../hooks/useTimeline';

const SPEEDS = [1, 2, 5, 10, 20];

export const Timeline: React.FC = () => {
  const { result, playbackSpeed, setPlaybackSpeed } = useSimulationStore();
  const { currentIndex, totalSteps, isPlaying, currentTimestamp, currentElapsed, seek, play, pause, stepForward, stepBack, reset } =
    useTimeline();

  if (!result) {
    return (
      <div className="h-16 border-t border-panel-border bg-panel-bg flex items-center justify-center text-text-muted text-xs">
        Timeline — Run a simulation first
      </div>
    );
  }

  const progress = totalSteps > 1 ? currentIndex / (totalSteps - 1) : 0;
  const timeStr = currentTimestamp
    ? `${currentTimestamp.replace('T', ' ').replace(/\.\d+$/, '')} UTC`
    : '— UTC';

  const elapsedSec = Math.max(0, Math.floor(currentElapsed));
  const hh = String(Math.floor(elapsedSec / 3600)).padStart(2, '0');
  const mm = String(Math.floor((elapsedSec % 3600) / 60)).padStart(2, '0');
  const ss = String(elapsedSec % 60).padStart(2, '0');
  const elapsedFormatted = `${hh}:${mm}:${ss}`;

  return (
    <div className="border-t border-panel-border bg-panel-bg px-4 py-2 flex items-center gap-4 select-none">
      {/* Playback buttons */}
      <div className="flex items-center gap-1">
        <button
          onClick={reset}
          title="Reset to start"
          className="text-text-secondary hover:text-text-primary p-1 text-sm"
        >⏮</button>
        <button
          onClick={stepBack}
          title="Step back"
          className="text-text-secondary hover:text-text-primary p-1 text-sm"
        >⏪</button>
        <button
          onClick={isPlaying ? pause : play}
          title={isPlaying ? 'Pause' : 'Play'}
          className="text-accent-cyan hover:text-cyan-300 p-1 text-lg"
        >
          {isPlaying ? '⏸' : '▶'}
        </button>
        <button
          onClick={stepForward}
          title="Step forward"
          className="text-text-secondary hover:text-text-primary p-1 text-sm"
        >⏩</button>
      </div>

      {/* Speed */}
      <div className="flex items-center gap-1">
        {SPEEDS.map(s => (
          <button
            key={s}
            onClick={() => setPlaybackSpeed(s)}
            className={`px-1.5 py-0.5 rounded text-xs border ${
              playbackSpeed === s
                ? 'border-accent-cyan text-accent-cyan'
                : 'border-panel-border text-text-muted hover:border-text-secondary'
            }`}
          >
            {s}×
          </button>
        ))}
      </div>

      {/* Slider */}
      <div className="flex-1 flex flex-col gap-0.5">
        <input
          type="range"
          min={0}
          max={totalSteps - 1}
          value={currentIndex}
          onChange={e => seek(parseInt(e.target.value))}
          className="w-full accent-cyan-400"
          style={{ accentColor: '#00d4ff' }}
        />
      </div>

      {/* Simulation Time display (Requirement 17) */}
      <div className="flex flex-col items-end min-w-[210px] pl-3 border-l border-panel-border text-right select-none">
        <div className="text-[10px] uppercase font-semibold text-text-muted tracking-wider leading-none">
          SIMULATION TIME
        </div>
        <div className="text-xs font-mono text-accent-cyan font-medium mt-1">
          {timeStr}
        </div>
        <div className="text-[11px] font-mono text-text-secondary flex gap-3 justify-end mt-0.5">
          <span>Elapsed: <span className="text-text-primary">{elapsedFormatted}</span></span>
          <span>Step: <span className="text-text-primary">{currentIndex + 1} / {totalSteps}</span></span>
        </div>
      </div>
    </div>
  );
};
