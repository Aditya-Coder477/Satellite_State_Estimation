// src/components/orbit3d/OrbitView3D.tsx
// 3D Orbit visualization using Plotly WebGL scatter3d.
// Earth sphere + reference trajectory (gold) + PI-STN trajectory (cyan).
// Animated by the global timeline index.

import React, { useMemo } from 'react';
import Plot from 'react-plotly.js';
import { useSimulationStore } from '../../store/simulationStore';

const R_EARTH = 6371; // km approximate sphere for visualization

function makeEarthSphere() {
  const N = 30;
  const theta = Array.from({ length: N }, (_, i) => (i / (N - 1)) * Math.PI);
  const phi = Array.from({ length: N }, (_, i) => (i / (N - 1)) * 2 * Math.PI);
  const x: number[][] = [], y: number[][] = [], z: number[][] = [];

  for (let i = 0; i < N; i++) {
    const row_x: number[] = [], row_y: number[] = [], row_z: number[] = [];
    for (let j = 0; j < N; j++) {
      row_x.push(R_EARTH * Math.sin(theta[i]) * Math.cos(phi[j]));
      row_y.push(R_EARTH * Math.sin(theta[i]) * Math.sin(phi[j]));
      row_z.push(R_EARTH * Math.cos(theta[i]));
    }
    x.push(row_x); y.push(row_y); z.push(row_z);
  }
  return { x, y, z };
}

const EARTH = makeEarthSphere();

export const OrbitView3D: React.FC = () => {
  const { result, currentIndex } = useSimulationStore();

  const traces = useMemo(() => {
    if (!result) return [];

    const ref = result.reference;
    const pred = result.prediction;

    const traces: any[] = [
      // Earth surface
      {
        type: 'surface',
        x: EARTH.x, y: EARTH.y, z: EARTH.z,
        colorscale: [['0', '#1a3a5c'], ['1', '#2a5980']],
        showscale: false,
        opacity: 0.85,
        hoverinfo: 'skip',
        name: 'Earth',
      },
      // Reference orbit (SGP4) — full trail
      {
        type: 'scatter3d',
        mode: 'lines',
        x: ref.x_km, y: ref.y_km, z: ref.z_km,
        line: { color: '#ffe66d', width: 2 },
        name: 'Reference (SGP4)',
        hovertemplate: 'SGP4<br>x=%{x:.1f} km<br>y=%{y:.1f} km<br>z=%{z:.1f} km<extra></extra>',
      },
      // Reference current position
      {
        type: 'scatter3d',
        mode: 'markers',
        x: [ref.x_km[currentIndex]],
        y: [ref.y_km[currentIndex]],
        z: [ref.z_km[currentIndex]],
        marker: { color: '#ffe66d', size: 6, symbol: 'circle' },
        name: 'SGP4 Now',
        hovertemplate: 'SGP4 position<extra></extra>',
      },
    ];

    if (pred) {
      traces.push(
        // PI-STN orbit
        {
          type: 'scatter3d',
          mode: 'lines',
          x: pred.x_km, y: pred.y_km, z: pred.z_km,
          line: { color: '#00d4ff', width: 2, dash: 'dot' },
          name: 'Prediction (PI-STN)',
          hovertemplate: 'PI-STN<br>x=%{x:.1f} km<br>y=%{y:.1f} km<br>z=%{z:.1f} km<extra></extra>',
          opacity: 0.8,
        },
        // PI-STN current position
        {
          type: 'scatter3d',
          mode: 'markers',
          x: [pred.x_km[currentIndex]],
          y: [pred.y_km[currentIndex]],
          z: [pred.z_km[currentIndex]],
          marker: { color: '#00d4ff', size: 6, symbol: 'diamond' },
          name: 'PI-STN Now',
          hovertemplate: 'PI-STN position<extra></extra>',
        },
      );
    }

    return traces;
  }, [result, currentIndex]);

  const layout = useMemo(() => ({
    paper_bgcolor: '#111827',
    plot_bgcolor: '#111827',
    scene: {
      bgcolor: '#0d1220',
      xaxis: { visible: false, range: [-50000, 50000] },
      yaxis: { visible: false, range: [-50000, 50000] },
      zaxis: { visible: false, range: [-50000, 50000] },
      aspectmode: 'cube' as const,
      camera: {
        eye: { x: 1.5, y: 1.5, z: 0.5 },
        up: { x: 0, y: 0, z: 1 },
      },
    },
    margin: { l: 0, r: 0, t: 0, b: 0 },
    legend: {
      bgcolor: 'rgba(17,24,39,0.8)',
      bordercolor: '#1e2d4a',
      borderwidth: 1,
      font: { color: '#e2e8f0', size: 11 },
    },
    showlegend: true,
  }), []);

  if (!result) {
    return (
      <div className="panel h-full flex items-center justify-center text-text-muted text-sm">
        <div className="text-center">
          <div className="text-4xl mb-2">🛰</div>
          <p>Run a simulation to view the 3D orbit</p>
        </div>
      </div>
    );
  }

  return (
    <div className="panel h-full flex flex-col overflow-hidden">
      <div className="px-3 py-2 border-b border-panel-border flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-widest text-text-muted">3D Orbit</h3>
        <span className="text-xs font-mono text-text-muted">GCRS frame</span>
      </div>
      <div className="flex-1 min-h-0">
        <Plot
          data={traces}
          layout={layout}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: '100%', height: '100%' }}
          useResizeHandler
        />
      </div>
    </div>
  );
};
