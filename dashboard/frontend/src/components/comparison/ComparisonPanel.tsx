// src/components/comparison/ComparisonPanel.tsx
// Research mode prediction comparison panel.
// Shows per-step error charts + summary metrics.
// All data from SimulationResult — no independent metric computation.

import React, { useState, useMemo } from 'react';
import Plot from 'react-plotly.js';
import { useSimulationStore } from '../../store/simulationStore';
import type { MetricsSummary } from '../../types';

const NA = 'Not available';

function fmtM(v: number | null | undefined, dec = 3, unit = ''): string {
  if (v == null || !isFinite(v)) return NA;
  return `${v.toFixed(dec)}${unit ? ' ' + unit : ''}`;
}

type Tab = 'position' | 'velocity' | 'orbital';

interface MetricCardProps { label: string; value: string; sub?: string }
const MetricCard: React.FC<MetricCardProps> = ({ label, value, sub }) => (
  <div className="bg-space-black border border-panel-border rounded p-2 min-w-0">
    <div className="text-xs text-text-muted mb-0.5 truncate">{label}</div>
    <div className="text-sm font-mono text-accent-cyan leading-tight">{value}</div>
    {sub && <div className="text-xs text-text-muted mt-0.5">{sub}</div>}
  </div>
);

export const ComparisonPanel: React.FC = () => {
  const { result, currentIndex } = useSimulationStore();
  const [activeTab, setActiveTab] = useState<Tab>('position');

  const baseLayout = useMemo(() => ({
    paper_bgcolor: '#111827',
    plot_bgcolor: '#0d1220',
    font: { color: '#94a3b8', size: 10 },
    margin: { l: 55, r: 20, t: 20, b: 40 },
    xaxis: {
      title: 'Elapsed Time (h)',
      color: '#64748b',
      gridcolor: '#1e2d4a',
      linecolor: '#1e2d4a',
    },
    legend: {
      bgcolor: 'rgba(17,24,39,0.8)',
      bordercolor: '#1e2d4a',
      borderwidth: 1,
      font: { color: '#e2e8f0', size: 10 },
    },
    showlegend: true,
  }), []);

  if (!result) {
    return (
      <div className="panel h-full flex flex-col overflow-hidden">
        <div className="px-3 py-2 border-b border-panel-border flex items-center justify-between">
          <h3 className="text-xs font-semibold uppercase tracking-widest text-text-muted">
            Prediction Comparison
          </h3>
          <span className="text-xs text-text-muted">
            REFERENCE: SGP4 &nbsp;|&nbsp; PREDICTION: PI-STN
          </span>
        </div>
        <div className="flex-1 flex items-center justify-center text-text-muted text-sm">
          <div className="text-center">
            <div className="text-4xl mb-2">📊</div>
            <p className="font-medium text-text-secondary">Prediction Comparison</p>
            <p className="text-xs mt-1 text-text-muted">Click &quot;Run Simulation&quot; in Compare mode to see metrics</p>
          </div>
        </div>
      </div>
    );
  }

  if (!result.comparison) {
    return (
      <div className="panel h-full flex flex-col overflow-hidden">
        <div className="px-3 py-2 border-b border-panel-border">
          <h3 className="text-xs font-semibold uppercase tracking-widest text-text-muted">Prediction Comparison</h3>
        </div>
        <div className="flex-1 flex items-center justify-center text-text-muted text-sm">
          <div className="text-center">
            <p className="text-accent-gold">ℹ PI-STN prediction not available</p>
            <p className="text-xs mt-1 text-text-muted">
              {result.satellite.ai_supported
                ? 'PI-STN inference failed — check backend logs'
                : 'This satellite does not have a trained PI-STN model'}
            </p>
          </div>
        </div>
      </div>
    );
  }

  const cmp = result.comparison;
  const m = cmp.metrics;
  const elapsed_h = result.elapsed_seconds.map(e => e / 3600);
  const cursor_h = result.elapsed_seconds[currentIndex] / 3600;

  const cursorShape = {
    type: 'line' as const,
    x0: cursor_h, x1: cursor_h,
    y0: 0, y1: 1,
    yref: 'paper' as const,
    line: { color: '#ffffff', width: 1, dash: 'dot' as const },
  };

  const posTraces: any[] = [
    {
      x: elapsed_h,
      y: cmp.position_error_km,
      type: 'scatter', mode: 'lines',
      name: 'Position Error ‖|r|‖',
      line: { color: '#ff6b6b', width: 1.5 },
    },
    {
      x: elapsed_h,
      y: cmp.err_x_km,
      type: 'scatter', mode: 'lines',
      name: 'Δx',
      line: { color: '#ff9999', width: 1, dash: 'dot' },
      visible: 'legendonly',
    },
    {
      x: elapsed_h,
      y: cmp.err_y_km,
      type: 'scatter', mode: 'lines',
      name: 'Δy',
      line: { color: '#ffcc99', width: 1, dash: 'dot' },
      visible: 'legendonly',
    },
    {
      x: elapsed_h,
      y: cmp.err_z_km,
      type: 'scatter', mode: 'lines',
      name: 'Δz',
      line: { color: '#ffff99', width: 1, dash: 'dot' },
      visible: 'legendonly',
    },
  ];

  const velTraces: any[] = [
    {
      x: elapsed_h,
      y: cmp.velocity_error_kms,
      type: 'scatter', mode: 'lines',
      name: 'Velocity Error ‖|v|‖',
      line: { color: '#4ecdc4', width: 1.5 },
    },
    {
      x: elapsed_h, y: cmp.err_vx_kms, type: 'scatter', mode: 'lines', name: 'Δvx',
      line: { color: '#80eeea', width: 1, dash: 'dot' }, visible: 'legendonly',
    },
    {
      x: elapsed_h, y: cmp.err_vy_kms, type: 'scatter', mode: 'lines', name: 'Δvy',
      line: { color: '#aafffd', width: 1, dash: 'dot' }, visible: 'legendonly',
    },
    {
      x: elapsed_h, y: cmp.err_vz_kms, type: 'scatter', mode: 'lines', name: 'Δvz',
      line: { color: '#ccffff', width: 1, dash: 'dot' }, visible: 'legendonly',
    },
  ];

  // Orbital element traces (reference vs prediction)
  const refOe = result.reference.orbital_elements;
  const predOe = result.prediction?.orbital_elements;
  const orbTraces: any[] = refOe && predOe ? [
    {
      x: elapsed_h, y: refOe.a_km, type: 'scatter', mode: 'lines',
      name: 'Ref a (km)', line: { color: '#ffe66d', width: 1.5 },
    },
    {
      x: elapsed_h, y: predOe.a_km, type: 'scatter', mode: 'lines',
      name: 'Pred a (km)', line: { color: '#00d4ff', width: 1.5, dash: 'dot' },
    },
  ] : [];

  const activeTraces = activeTab === 'position' ? posTraces
    : activeTab === 'velocity' ? velTraces
    : orbTraces;

  const activeYLabel = activeTab === 'position' ? 'Error (km)'
    : activeTab === 'velocity' ? 'Error (km/s)'
    : 'Semi-major axis a (km)';

  return (
    <div className="panel h-full flex flex-col overflow-hidden">
      <div className="px-3 py-2 border-b border-panel-border flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-widest text-text-muted">
          Prediction Comparison
        </h3>
        <span className="text-xs text-text-muted">
          REFERENCE: SGP4 &nbsp;|&nbsp; PREDICTION: PI-STN
        </span>
      </div>

      {/* Metric cards */}
      <div className="px-3 py-2 grid grid-cols-3 gap-2 border-b border-panel-border">
        <MetricCard label="Pos RMSE" value={fmtM(m.position_rmse_km, 3, 'km')} />
        <MetricCard label="Vel RMSE" value={fmtM(m.velocity_rmse_kms, 5, 'km/s')} />
        <MetricCard label="Max Pos Error" value={fmtM(m.position_max_error_km, 3, 'km')} />
        <MetricCard label="Cumul. Drift" value={fmtM(m.cumulative_drift_km, 2, 'km')} />
        <MetricCard label="Error Growth" value={fmtM(m.error_growth_rate_kmps, 5, 'km/s/s')} />
        <MetricCard label="Physics Res." value={
          m.physics_available
            ? fmtM(m.physics_residual_mean, 3, 'km/s²')
            : 'Not available'
        } />
      </div>

      {/* Tab switcher */}
      <div className="flex gap-0 border-b border-panel-border">
        {(['position', 'velocity', 'orbital'] as Tab[]).map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`flex-1 py-1.5 text-xs font-medium transition-colors border-r border-panel-border last:border-0 ${
              activeTab === tab
                ? 'bg-space-black text-accent-cyan'
                : 'text-text-muted hover:text-text-primary'
            }`}
          >
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </div>

      {/* Chart */}
      <div className="flex-1 min-h-0">
        {activeTraces.length > 0 ? (
          <Plot
            data={activeTraces}
            layout={{
              ...baseLayout,
              yaxis: {
                title: activeYLabel,
                color: '#64748b',
                gridcolor: '#1e2d4a',
                linecolor: '#1e2d4a',
              },
              shapes: [cursorShape],
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%', height: '100%' }}
            useResizeHandler
          />
        ) : (
          <div className="h-full flex items-center justify-center text-text-muted text-xs">
            No orbital element data available
          </div>
        )}
      </div>

      {/* Orbital element metrics */}
      {m.a_km_rmse != null && (
        <div className="px-3 py-1.5 border-t border-panel-border grid grid-cols-3 gap-2">
          <MetricCard label="Δa RMSE" value={fmtM(m.a_km_rmse, 3, 'km')} />
          <MetricCard label="Δe RMSE" value={fmtM(m.e_rmse, 6)} />
          <MetricCard label="Δi RMSE" value={fmtM(m.i_deg_rmse, 4, '°')} />
        </div>
      )}
    </div>
  );
};
