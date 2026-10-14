// src/components/controls/SimulationControls.tsx
import React, { useEffect } from 'react';
import { useSimulationStore } from '../../store/simulationStore';
import { api } from '../../services/api';
import type { SatelliteInfo } from '../../types';

const DURATIONS = [6, 12, 24, 48, 72];
const STEPS = [30, 60, 120, 300];

export const SimulationControls: React.FC = () => {
  const {
    satellites, setSatellites,
    selectedSatelliteId, setSelectedSatellite,
    durationHours, setDurationHours,
    stepSeconds, setStepSeconds,
    simulationMode, setSimulationMode,
    setResult, setLoading, setError, isLoading,
  } = useSimulationStore();

  // Load satellite list on mount
  useEffect(() => {
    api.listSatellites()
      .then(setSatellites)
      .catch(err => console.error('Failed to load satellites:', err));
  }, []);

  const selectedSat = satellites.find(s => s.id === selectedSatelliteId);
  const aiAvailable = selectedSat?.checkpoint_available && selectedSat?.ai_supported;

  const handleRun = async () => {
    setLoading(true);
    setError(null);
    try {
      const req = {
        satellite_id: selectedSatelliteId,
        duration_hours: durationHours,
        step_seconds: stepSeconds,
      };
      let result;
      if (simulationMode === 'compare' && aiAvailable) {
        result = await api.runCompare(req);
      } else {
        result = await api.runSgp4(req);
      }
      setResult(result);
    } catch (err: any) {
      const msg = err?.response?.data?.detail ?? err?.message ?? 'Simulation failed';
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="panel p-4 space-y-4">
      <h2 className="text-xs font-semibold uppercase tracking-widest text-text-muted">
        Simulation Controls
      </h2>

      {/* Satellite selector */}
      <div>
        <label className="block text-xs text-text-secondary mb-1">Satellite</label>
        <select
          value={selectedSatelliteId}
          onChange={e => {
            const nextId = e.target.value;
            setSelectedSatellite(nextId);
            const targetSat = satellites.find(s => s.id === nextId);
            if (targetSat && !targetSat.checkpoint_available) {
              setSimulationMode('sgp4');
            }
          }}
          className="w-full bg-space-black border border-panel-border rounded px-3 py-2 text-sm text-text-primary focus:border-accent-cyan focus:outline-none"
        >
          {satellites.map(sat => (
            <option key={sat.id} value={sat.id}>
              {sat.display_name} ({sat.operator})
              {!sat.checkpoint_available ? ' — No AI' : ''}
            </option>
          ))}
          {satellites.length === 0 && (
            <option value="INSAT3D">Loading...</option>
          )}
        </select>

        {/* Satellite metadata */}
        {selectedSat && (
          <div className="mt-2 grid grid-cols-2 gap-1 text-xs font-mono">
            <span className="text-text-muted">NORAD</span>
            <span className="text-text-primary">{selectedSat.norad_id}</span>
            <span className="text-text-muted">Orbit</span>
            <span className="text-text-primary">{selectedSat.orbit_type}</span>
            <span className="text-text-muted">AI Model</span>
            <span className={aiAvailable ? 'text-accent-green' : 'text-accent-red'}>
              {aiAvailable ? '✓ Available' : '✗ Not available'}
            </span>
          </div>
        )}
      </div>

      {/* Mode */}
      <div>
        <label className="block text-xs text-text-secondary mb-1">Mode</label>
        <div className="flex gap-2">
          <button
            onClick={() => setSimulationMode('sgp4')}
            className={`flex-1 py-1.5 rounded text-xs font-medium border transition-colors ${
              simulationMode === 'sgp4'
                ? 'bg-accent-cyan text-space-black border-accent-cyan'
                : 'border-panel-border text-text-secondary hover:border-accent-cyan'
            }`}
          >
            SGP4 Only
          </button>
          <button
            onClick={() => setSimulationMode('compare')}
            disabled={!aiAvailable}
            title={!aiAvailable ? 'PI-STN not available for this satellite' : ''}
            className={`flex-1 py-1.5 rounded text-xs font-medium border transition-colors ${
              simulationMode === 'compare' && aiAvailable
                ? 'bg-accent-gold text-space-black border-accent-gold'
                : aiAvailable
                ? 'border-panel-border text-text-secondary hover:border-accent-gold'
                : 'border-panel-border text-text-muted opacity-40 cursor-not-allowed'
            }`}
          >
            Compare
          </button>
        </div>
        {simulationMode === 'compare' && aiAvailable && (
          <p className="text-xs text-text-muted mt-1">
            REFERENCE: SGP4 &nbsp;|&nbsp; PREDICTION: PI-STN
          </p>
        )}
      </div>

      {/* Duration */}
      <div>
        <label className="block text-xs text-text-secondary mb-1">
          Duration: <span className="font-mono text-accent-cyan">{durationHours}h</span>
        </label>
        <div className="flex flex-wrap gap-1">
          {DURATIONS.map(d => (
            <button
              key={d}
              onClick={() => setDurationHours(d)}
              className={`px-2 py-1 rounded text-xs border transition-colors ${
                durationHours === d
                  ? 'bg-accent-cyan text-space-black border-accent-cyan'
                  : 'border-panel-border text-text-secondary hover:border-accent-cyan'
              }`}
            >
              {d}h
            </button>
          ))}
        </div>
      </div>

      {/* Step */}
      <div>
        <label className="block text-xs text-text-secondary mb-1">
          Step: <span className="font-mono text-accent-cyan">{stepSeconds}s</span>
        </label>
        <div className="flex flex-wrap gap-1">
          {STEPS.map(s => (
            <button
              key={s}
              onClick={() => setStepSeconds(s)}
              className={`px-2 py-1 rounded text-xs border transition-colors ${
                stepSeconds === s
                  ? 'bg-accent-cyan text-space-black border-accent-cyan'
                  : 'border-panel-border text-text-secondary hover:border-accent-cyan'
              }`}
            >
              {s}s
            </button>
          ))}
        </div>
      </div>

      {/* Run button */}
      <button
        onClick={handleRun}
        disabled={isLoading}
        className="w-full py-2.5 rounded font-semibold text-sm transition-all bg-accent-cyan text-space-black hover:bg-cyan-300 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {isLoading ? (
          <span className="flex items-center justify-center gap-2">
            <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
            </svg>
            Running…
          </span>
        ) : (
          '▶  Run Simulation'
        )}
      </button>

      {/* Est. steps hint */}
      <p className="text-xs text-text-muted text-center font-mono">
        ~{Math.ceil((durationHours * 3600) / stepSeconds + 1).toLocaleString()} steps
        {simulationMode === 'compare' && aiAvailable && ' (SGP4 + PI-STN)'}
      </p>
    </div>
  );
};
