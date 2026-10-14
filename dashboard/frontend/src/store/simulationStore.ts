// src/store/simulationStore.ts
// Zustand global state store.
// All four dashboard views subscribe to the same store — no independent timing.

import { create } from 'zustand';
import type { SimulationResult, DashboardMode, SimulationMode, SatelliteInfo } from '../types';

interface SimulationStore {
  // ── Simulation data ─────────────────────────────────────────────
  result: SimulationResult | null;
  isLoading: boolean;
  error: string | null;

  // ── Control parameters ──────────────────────────────────────────
  selectedSatelliteId: string;
  durationHours: number;
  stepSeconds: number;
  simulationMode: SimulationMode;   // 'sgp4' | 'compare'
  dashboardMode: DashboardMode;     // 'explorer' | 'research'

  // ── Satellite catalog ──────────────────────────────────────────
  satellites: SatelliteInfo[];

  // ── Timeline ───────────────────────────────────────────────────
  currentIndex: number;     // index into timestamps[], 0-based
  isPlaying: boolean;
  playbackSpeed: number;    // multiplier: 1, 2, 5, 10

  // ── Actions ────────────────────────────────────────────────────
  setResult: (result: SimulationResult) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setCurrentIndex: (idx: number) => void;
  setPlaying: (playing: boolean) => void;
  setPlaybackSpeed: (speed: number) => void;
  setSelectedSatellite: (id: string) => void;
  setDurationHours: (h: number) => void;
  setStepSeconds: (s: number) => void;
  setSimulationMode: (mode: SimulationMode) => void;
  setDashboardMode: (mode: DashboardMode) => void;
  setSatellites: (sats: SatelliteInfo[]) => void;
  reset: () => void;
}

const getInitialDashboardMode = (): DashboardMode => {
  try {
    const saved = localStorage.getItem('satellite_orbit_lab_mode');
    if (saved === 'explorer' || saved === 'research') return saved;
  } catch {}
  return 'research';
};

const initialState = {
  result: null,
  isLoading: false,
  error: null,
  selectedSatelliteId: 'INSAT3D',   // default per spec
  durationHours: 24,
  stepSeconds: 60,
  simulationMode: 'compare' as SimulationMode,   // default per spec
  dashboardMode: getInitialDashboardMode(),
  satellites: [],
  currentIndex: 0,
  isPlaying: false,
  playbackSpeed: 1,
};

export const useSimulationStore = create<SimulationStore>((set) => ({
  ...initialState,

  setResult: (result) => set({ result, currentIndex: 0, error: null }),
  setLoading: (isLoading) => set({ isLoading }),
  setError: (error) => set({ error }),
  setCurrentIndex: (currentIndex) => set({ currentIndex }),
  setPlaying: (isPlaying) => set({ isPlaying }),
  setPlaybackSpeed: (playbackSpeed) => set({ playbackSpeed }),
  setSelectedSatellite: (id) => set({ selectedSatelliteId: id, result: null, currentIndex: 0 }),
  setDurationHours: (durationHours) => set({ durationHours }),
  setStepSeconds: (stepSeconds) => set({ stepSeconds }),
  setSimulationMode: (simulationMode) => set({ simulationMode }),
  setDashboardMode: (dashboardMode) => {
    try {
      localStorage.setItem('satellite_orbit_lab_mode', dashboardMode);
    } catch {}
    set({ dashboardMode });
  },
  setSatellites: (satellites) => set({ satellites }),
  reset: () => set({ ...initialState }),
}));
