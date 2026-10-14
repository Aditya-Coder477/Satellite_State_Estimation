// src/services/api.ts
// Axios-based API client for the Satellite Orbit Lab backend.

import axios from 'axios';
import type { SatelliteInfo, SimulationRequest, SimulationResult } from '../types';

const BASE = '/api';

const client = axios.create({
  baseURL: BASE,
  timeout: 300_000, // 5 min — PI-STN rollout can be slow on first run
});

export const api = {
  /** List all registered satellites */
  listSatellites: (): Promise<SatelliteInfo[]> =>
    client.get('/satellites').then(r => r.data),

  /** Get details for a specific satellite */
  getSatellite: (id: string): Promise<SatelliteInfo & { ground_stations: any[] }> =>
    client.get(`/satellites/${id}`).then(r => r.data),

  /** Run SGP4-only simulation */
  runSgp4: (req: SimulationRequest): Promise<SimulationResult> =>
    client.post('/simulation/sgp4', req).then(r => r.data),

  /** Run SGP4 + PI-STN comparison */
  runCompare: (req: SimulationRequest): Promise<SimulationResult> =>
    client.post('/evaluation/compare', req).then(r => r.data),

  /** Health check */
  health: () => client.get('/health').then(r => r.data),
};
