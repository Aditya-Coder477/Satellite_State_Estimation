// src/components/layout/Header.tsx
import React from 'react';
import { useSimulationStore } from '../../store/simulationStore';

export const Header: React.FC = () => {
  const { dashboardMode, setDashboardMode } = useSimulationStore();

  return (
    <header className="flex items-center justify-between px-6 py-3 border-b border-panel-border bg-panel-bg select-none">
      {/* Brand */}
      <div className="flex items-center gap-3">
        <span className="text-accent-cyan text-2xl">🛰</span>
        <div>
          <h1 className="text-lg font-semibold text-text-primary tracking-wide">
            SATELLITE ORBIT LAB
          </h1>
          <p className="text-xs text-text-muted font-mono">
            Physics-Informed State Transition Network
          </p>
        </div>
      </div>

      {/* Mode switcher */}
      <div className="flex items-center gap-2 bg-space-black rounded-lg p-1 border border-panel-border">
        <button
          onClick={() => setDashboardMode('explorer')}
          className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors ${
            dashboardMode === 'explorer'
              ? 'bg-accent-cyan text-space-black'
              : 'text-text-secondary hover:text-text-primary'
          }`}
        >
          Explorer
        </button>
        <button
          onClick={() => setDashboardMode('research')}
          className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors ${
            dashboardMode === 'research'
              ? 'bg-accent-gold text-space-black'
              : 'text-text-secondary hover:text-text-primary'
          }`}
        >
          Research
        </button>
      </div>

      {/* Status */}
      <div className="flex items-center gap-2">
        <span className="inline-block w-2 h-2 rounded-full bg-accent-green animate-pulse" />
        <span className="text-xs text-text-muted font-mono">API CONNECTED</span>
      </div>
    </header>
  );
};
