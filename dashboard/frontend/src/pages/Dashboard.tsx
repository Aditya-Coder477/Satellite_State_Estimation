// src/pages/Dashboard.tsx
// Main dashboard layout.
// Explorer mode: 3 panels (3D Orbit, Ground Track, State Info)
// Research mode: 4 panels (above + Prediction Comparison)

import React from 'react';
import { useSimulationStore } from '../store/simulationStore';
import { SimulationControls } from '../components/controls/SimulationControls';
import { OrbitView3D } from '../components/orbit3d/OrbitView3D';
import { GroundTrack } from '../components/groundTrack/GroundTrack';
import { StatePanel } from '../components/statePanel/StatePanel';
import { ComparisonPanel } from '../components/comparison/ComparisonPanel';
import { Timeline } from '../components/timeline/Timeline';

export const Dashboard: React.FC = () => {
  const { dashboardMode, error } = useSimulationStore();

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Error banner */}
      {error && (
        <div className="bg-red-900/40 border-b border-red-700 px-4 py-2 text-sm text-accent-red">
          ⚠ {error}
        </div>
      )}

      <div className="flex flex-1 overflow-hidden">
        {/* Left sidebar — controls */}
        <aside className="w-64 min-w-[16rem] border-r border-panel-border overflow-y-auto p-3 bg-panel-bg">
          <SimulationControls />
        </aside>

        {/* Main content */}
        <main className="flex-1 flex flex-col overflow-hidden">
          {/* Main 4-panel grid — 3D Orbit, Ground Track, State Info, Prediction Comparison */}
          <div
            className={`flex-1 grid grid-cols-2 gap-2 p-2 overflow-hidden ${
              dashboardMode === 'explorer'
                ? 'grid-rows-[1.2fr_0.8fr]'
                : 'grid-rows-2'
            }`}
          >
            <div className="min-h-0">
              <OrbitView3D />
            </div>
            <div className="min-h-0">
              <GroundTrack />
            </div>
            <div className="min-h-0">
              <StatePanel />
            </div>
            <div className="min-h-0">
              <ComparisonPanel />
            </div>
          </div>

          {/* Timeline — always shown at bottom */}
          <Timeline />
        </main>
      </div>
    </div>
  );
};
