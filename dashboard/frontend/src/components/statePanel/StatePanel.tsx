// src/components/statePanel/StatePanel.tsx
// Telemetry panel — shows real-time state at the current timeline index.
// All values come from the SimulationResult — no independent computation.

import React from 'react';
import { useSimulationStore } from '../../store/simulationStore';

const NA = 'Not available';

function fmt(v: number | null | undefined, decimals = 3, unit = ''): string {
  if (v == null || !isFinite(v)) return NA;
  return `${v.toFixed(decimals)}${unit ? ' ' + unit : ''}`;
}

interface RowProps { label: string; value: string; color?: string }
const Row: React.FC<RowProps> = ({ label, value, color = 'text-accent-cyan' }) => (
  <div className="flex justify-between items-baseline py-0.5 border-b border-panel-border last:border-0">
    <span className="text-xs text-text-muted">{label}</span>
    <span className={`text-xs font-mono ${color}`}>{value}</span>
  </div>
);

interface SectionProps { title: string; children: React.ReactNode }
const Section: React.FC<SectionProps> = ({ title, children }) => (
  <div className="mb-3">
    <h4 className="text-xs text-text-muted uppercase tracking-widest mb-1 border-b border-panel-border pb-1">
      {title}
    </h4>
    {children}
  </div>
);

export const StatePanel: React.FC = () => {
  const { result, currentIndex } = useSimulationStore();

  if (!result) {
    return (
      <div className="panel h-full flex items-center justify-center text-text-muted text-sm">
        <div className="text-center">
          <div className="text-4xl mb-2">📡</div>
          <p>State information</p>
          <p className="text-xs mt-1">Run a simulation to view telemetry</p>
        </div>
      </div>
    );
  }

  const ref = result.reference;
  const pred = result.prediction;
  const ts = result.timestamps[currentIndex] ?? '';
  const el = result.elapsed_seconds[currentIndex] ?? 0;
  const oe = ref.orbital_elements;
  const predOe = pred?.orbital_elements;

  const refRow = (label: string, val: number | null | undefined, dec = 3, unit = '') =>
    <Row key={label} label={label} value={fmt(val, dec, unit)} color="text-accent-gold" />;

  const predRow = (label: string, val: number | null | undefined, dec = 3, unit = '') =>
    <Row key={label} label={label} value={fmt(val, dec, unit)} color="text-accent-cyan" />;

  return (
    <div className="panel h-full flex flex-col overflow-hidden">
      <div className="px-3 py-2 border-b border-panel-border">
        <h3 className="text-xs font-semibold uppercase tracking-widest text-text-muted">State Information</h3>
      </div>

      <div className="flex-1 overflow-y-auto px-3 py-2 space-y-0 text-xs">
        {/* Timestamp */}
        <Section title="Time">
          <Row label="UTC" value={ts.replace('T', ' ').replace(/\.\d+$/, '')} />
          <Row label="Elapsed" value={`${el.toFixed(0)} s (${(el / 3600).toFixed(2)} h)`} />
        </Section>

        {/* Position */}
        <Section title="Position — GCRS (km)">
          {refRow('Ref X', ref.x_km[currentIndex], 2)}
          {refRow('Ref Y', ref.y_km[currentIndex], 2)}
          {refRow('Ref Z', ref.z_km[currentIndex], 2)}
          {pred && predRow('Pred X', pred.x_km[currentIndex], 2)}
          {pred && predRow('Pred Y', pred.y_km[currentIndex], 2)}
          {pred && predRow('Pred Z', pred.z_km[currentIndex], 2)}
        </Section>

        {/* Velocity */}
        <Section title="Velocity — GCRS (km/s)">
          {refRow('Ref Vx', ref.vx_kms[currentIndex], 5)}
          {refRow('Ref Vy', ref.vy_kms[currentIndex], 5)}
          {refRow('Ref Vz', ref.vz_kms[currentIndex], 5)}
          {pred && predRow('Pred Vx', pred.vx_kms[currentIndex], 5)}
          {pred && predRow('Pred Vy', pred.vy_kms[currentIndex], 5)}
          {pred && predRow('Pred Vz', pred.vz_kms[currentIndex], 5)}
        </Section>

        {/* Geographic */}
        <Section title="Geographic (WGS-84)">
          {refRow('Ref Lat', ref.lat_deg[currentIndex], 4, '°')}
          {refRow('Ref Lon', ref.lon_deg[currentIndex], 4, '°')}
          {refRow('Ref Alt', ref.alt_km[currentIndex], 2, 'km')}
          {pred && predRow('Pred Lat', pred.lat_deg[currentIndex], 4, '°')}
          {pred && predRow('Pred Lon', pred.lon_deg[currentIndex], 4, '°')}
          {pred && predRow('Pred Alt', pred.alt_km[currentIndex], 2, 'km')}
        </Section>

        {/* Orbital elements */}
        <Section title="Orbital Elements (GCRS)">
          {refRow('Ref a', oe?.a_km?.[currentIndex], 2, 'km')}
          {refRow('Ref e', oe?.e?.[currentIndex], 6)}
          {refRow('Ref i', oe?.i_deg?.[currentIndex], 4, '°')}
          {refRow('Ref RAAN', oe?.raan_deg?.[currentIndex], 4, '°')}
          {refRow('Ref ω', oe?.argp_deg?.[currentIndex], 4, '°')}
          {refRow('Ref ν', oe?.nu_deg?.[currentIndex], 4, '°')}
          {pred && predRow('Pred a', predOe?.a_km?.[currentIndex], 2, 'km')}
          {pred && predRow('Pred e', predOe?.e?.[currentIndex], 6)}
          {pred && predRow('Pred i', predOe?.i_deg?.[currentIndex], 4, '°')}
        </Section>

        {/* Model status */}
        <Section title="Model Status">
          <Row label="Satellite" value={result.satellite.display_name} />
          <Row label="Operator" value={result.satellite.operator} />
          <Row label="NORAD" value={result.satellite.norad_id.toString()} />
          <Row label="PI-STN" value={result.metadata.pistn_available ? '✓ Active' : '✗ Not available'}
            color={result.metadata.pistn_available ? 'text-accent-green' : 'text-text-muted'} />
          <Row label="Mode" value={result.metadata.mode.toUpperCase()} />
          <Row label="Δt" value={`${result.metadata.delta_t_seconds}s`} />
          <Row label="Steps" value={result.metadata.n_steps.toLocaleString()} />
        </Section>
      </div>
    </div>
  );
};
