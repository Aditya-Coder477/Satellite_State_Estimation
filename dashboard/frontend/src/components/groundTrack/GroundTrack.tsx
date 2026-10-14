// src/components/groundTrack/GroundTrack.tsx
// Ground track visualization using Leaflet.
// Renders reference ground track (SGP4), prediction ground track (PI-STN),
// current satellite marker, start/end markers, and ground stations.
// Synchronized to the global Zustand timeline index.

import React, { useEffect, useRef, useState, useMemo, useCallback } from 'react';
import { useSimulationStore } from '../../store/simulationStore';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Fix Leaflet marker icons in Vite bundler
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
});

// Tile provider configurations (Requirement 5)
type TileProviderKey = 'dark' | 'osm' | 'satellite';

interface TileProvider {
  name: string;
  url: string;
  attribution: string;
  subdomains?: string;
  maxZoom: number;
  className?: string;
}

const TILE_PROVIDERS: Record<TileProviderKey, TileProvider> = {
  dark: {
    name: 'Dark Matter',
    url: 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>',
    subdomains: 'abcd',
    maxZoom: 19,
  },
  osm: {
    name: 'Street Map',
    url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
    maxZoom: 19,
    className: 'osm-dark-filter',
  },
  satellite: {
    name: 'Satellite Imagery',
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attribution: '&copy; Esri, Earthstar Geographics',
    maxZoom: 18,
  },
};

/**
 * Normalizes longitude to [-180, +180] and splits trajectory when crossing the antimeridian.
 * Prevents horizontal artifacts across the world map (Requirement 11).
 */
function splitTrajectoryAtAntimeridian(lats: number[], lons: number[]): L.LatLngTuple[][] {
  if (!lats || !lons || lats.length === 0) return [];
  const segments: L.LatLngTuple[][] = [];
  let currentSegment: L.LatLngTuple[] = [];

  for (let i = 0; i < lats.length; i++) {
    const lat = lats[i];
    let lon = lons[i];

    // Normalize longitude to [-180, 180]
    lon = ((((lon + 180) % 360) + 360) % 360) - 180;

    if (currentSegment.length > 0) {
      const prevLon = currentSegment[currentSegment.length - 1][1];
      // Crossing meridian jump
      if (Math.abs(lon - prevLon) > 180) {
        segments.push(currentSegment);
        currentSegment = [];
      }
    }
    currentSegment.push([lat, lon]);
  }
  if (currentSegment.length > 0) {
    segments.push(currentSegment);
  }
  return segments;
}

export const GroundTrack: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<L.Map | null>(null);
  const tileLayerRef = useRef<L.TileLayer | null>(null);

  // Layer references for dynamic updates
  const layersRef = useRef<{
    refLine?: L.Polyline;
    predLine?: L.Polyline;
    refMarker?: L.CircleMarker;
    predMarker?: L.CircleMarker;
    startMarker?: L.CircleMarker;
    endMarker?: L.CircleMarker;
    stationGroup?: L.LayerGroup;
  }>({});

  const { result, currentIndex, simulationMode } = useSimulationStore();

  const [activeTile, setActiveTile] = useState<TileProviderKey>('dark');
  const [showStations, setShowStations] = useState<boolean>(true);
  const [mapInitialized, setMapInitialized] = useState<boolean>(false);

  // ── 1. Initialize Leaflet Map once on container mount ───────────────────
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    try {
      const map = L.map(containerRef.current, {
        center: [10, 20],
        zoom: 2,
        minZoom: 1,
        maxZoom: 18,
        zoomControl: true,
        attributionControl: false,
      });

      // Add default tile layer
      const provider = TILE_PROVIDERS[activeTile];
      const tileLayer = L.tileLayer(provider.url, {
        attribution: provider.attribution,
        subdomains: provider.subdomains || 'abc',
        maxZoom: provider.maxZoom,
        className: provider.className || '',
      }).addTo(map);

      tileLayerRef.current = tileLayer;
      mapRef.current = map;
      setMapInitialized(true);

      // Force layout invalidation after DOM paint
      setTimeout(() => {
        map.invalidateSize();
      }, 150);
    } catch (err) {
      console.error('[GroundTrack] Map initialization error:', err);
    }

    return () => {
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
        setMapInitialized(false);
      }
    };
  }, []);

  // ── 2. Handle container resizing automatically ──────────────────────────
  useEffect(() => {
    if (!containerRef.current || !mapRef.current) return;

    const ro = new ResizeObserver(() => {
      mapRef.current?.invalidateSize();
    });
    ro.observe(containerRef.current);

    return () => {
      ro.disconnect();
    };
  }, [mapInitialized]);

  // ── 3. Switch Tile Layer when activeTile changes ────────────────────────
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    if (tileLayerRef.current) {
      map.removeLayer(tileLayerRef.current);
    }

    const provider = TILE_PROVIDERS[activeTile];
    const newLayer = L.tileLayer(provider.url, {
      attribution: provider.attribution,
      subdomains: provider.subdomains || 'abc',
      maxZoom: provider.maxZoom,
      className: provider.className || '',
    }).addTo(map);

    tileLayerRef.current = newLayer;
  }, [activeTile]);

  // ── Helper to build popup HTML for current satellite ────────────────────
  const buildCurrentPopupContent = useCallback((index: number) => {
    if (!result) return '';
    const ref = result.reference;
    const pred = result.prediction;
    const ts = result.timestamps[index] || '';
    const utcFormatted = ts.replace('T', ' ').replace(/\.\d+$/, '') + ' UTC';

    let html = `
      <div style="font-family: Inter, sans-serif; font-size: 11px; min-width: 170px;">
        <div style="font-weight: 600; color: #00d4ff; font-size: 12px; margin-bottom: 4px; border-bottom: 1px solid #1e2d4a; padding-bottom: 3px;">
          🛰 ${result.satellite.display_name}
        </div>
        <div style="margin-bottom: 2px; color: #94a3b8;">
          <span style="color: #64748b;">Time:</span> ${utcFormatted}
        </div>
        <div style="margin-bottom: 2px; color: #ffe66d; font-family: monospace;">
          <span style="color: #64748b;">Ref Lat:</span> ${ref.lat_deg[index]?.toFixed(4)}°
        </div>
        <div style="margin-bottom: 2px; color: #ffe66d; font-family: monospace;">
          <span style="color: #64748b;">Ref Lon:</span> ${ref.lon_deg[index]?.toFixed(4)}°
        </div>
        <div style="margin-bottom: 2px; color: #ffe66d; font-family: monospace;">
          <span style="color: #64748b;">Ref Alt:</span> ${ref.alt_km[index]?.toFixed(2)} km
        </div>
    `;

    if (pred && pred.lat_deg[index] != null) {
      html += `
        <div style="margin-top: 4px; padding-top: 3px; border-top: 1px solid #1e2d4a;">
          <div style="color: #00d4ff; font-family: monospace;">
            <span style="color: #64748b;">Pred Lat:</span> ${pred.lat_deg[index]?.toFixed(4)}°
          </div>
          <div style="color: #00d4ff; font-family: monospace;">
            <span style="color: #64748b;">Pred Lon:</span> ${pred.lon_deg[index]?.toFixed(4)}°
          </div>
          <div style="color: #00d4ff; font-family: monospace;">
            <span style="color: #64748b;">Pred Alt:</span> ${pred.alt_km[index]?.toFixed(2)} km
          </div>
        </div>
      `;
    }

    html += `
        <div style="margin-top: 4px; font-size: 10px; color: #64748b;">
          Mode: <span style="color: #e2e8f0;">${result.metadata.mode.toUpperCase()}</span>
        </div>
      </div>
    `;
    return html;
  }, [result]);

  // ── 4. Re-draw Trajectories, Markers & Ground Stations on Result Change ──
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !result) return;

    // Clear previous layers
    if (layersRef.current.refLine) map.removeLayer(layersRef.current.refLine);
    if (layersRef.current.predLine) map.removeLayer(layersRef.current.predLine);
    if (layersRef.current.refMarker) map.removeLayer(layersRef.current.refMarker);
    if (layersRef.current.predMarker) map.removeLayer(layersRef.current.predMarker);
    if (layersRef.current.startMarker) map.removeLayer(layersRef.current.startMarker);
    if (layersRef.current.endMarker) map.removeLayer(layersRef.current.endMarker);
    if (layersRef.current.stationGroup) map.removeLayer(layersRef.current.stationGroup);

    layersRef.current = {};

    const ref = result.reference;
    const pred = result.prediction;

    // Diagnostic logging (Requirement 14)
    console.debug('[GroundTrack] Diagnostics:', {
      referencePoints: ref.lat_deg?.length,
      predictionPoints: pred ? pred.lat_deg?.length : 0,
      firstRefCoord: ref.lat_deg?.length ? [ref.lat_deg[0], ref.lon_deg[0]] : null,
      lastRefCoord: ref.lat_deg?.length ? [ref.lat_deg[ref.lat_deg.length - 1], ref.lon_deg[ref.lon_deg.length - 1]] : null,
      firstPredCoord: pred && pred.lat_deg?.length ? [pred.lat_deg[0], pred.lon_deg[0]] : null,
      lastPredCoord: pred && pred.lat_deg?.length ? [pred.lat_deg[pred.lat_deg.length - 1], pred.lon_deg[pred.lon_deg.length - 1]] : null,
      hasNaN: ref.lat_deg?.some(v => isNaN(v)) || ref.lon_deg?.some(v => isNaN(v)),
    });

    if (!ref.lat_deg || ref.lat_deg.length === 0) {
      return;
    }

    // A. Reference Polyline with antimeridian splitting (Gold)
    const refSegments = splitTrajectoryAtAntimeridian(ref.lat_deg, ref.lon_deg);
    const refLine = L.polyline(refSegments, {
      color: '#ffe66d',
      weight: 2.5,
      opacity: 0.95,
      lineCap: 'round',
      lineJoin: 'round',
    }).addTo(map);
    layersRef.current.refLine = refLine;

    // B. Prediction Polyline (Cyan dashed)
    if (pred && pred.lat_deg && pred.lat_deg.length > 0) {
      const predSegments = splitTrajectoryAtAntimeridian(pred.lat_deg, pred.lon_deg);
      const predLine = L.polyline(predSegments, {
        color: '#00d4ff',
        weight: 2,
        opacity: 0.85,
        dashArray: '5 5',
        lineCap: 'round',
        lineJoin: 'round',
      }).addTo(map);
      layersRef.current.predLine = predLine;
    }

    // C. Start & End Markers (Requirement 4)
    const startLat = ref.lat_deg[0];
    const startLon = ((((ref.lon_deg[0] + 180) % 360) + 360) % 360) - 180;
    const startMarker = L.circleMarker([startLat, startLon], {
      radius: 5,
      color: '#4ecdc4',
      fillColor: '#4ecdc4',
      fillOpacity: 0.9,
      weight: 1.5,
    }).bindTooltip('Start (Epoch)', { direction: 'top' }).addTo(map);
    layersRef.current.startMarker = startMarker;

    const lastIdx = ref.lat_deg.length - 1;
    const endLat = ref.lat_deg[lastIdx];
    const endLon = ((((ref.lon_deg[lastIdx] + 180) % 360) + 360) % 360) - 180;
    const endMarker = L.circleMarker([endLat, endLon], {
      radius: 5,
      color: '#ff6b6b',
      fillColor: '#ff6b6b',
      fillOpacity: 0.9,
      weight: 1.5,
    }).bindTooltip('End of Horizon', { direction: 'bottom' }).addTo(map);
    layersRef.current.endMarker = endMarker;

    // D. Current Position Marker (Reference — Gold)
    const curLat = ref.lat_deg[0];
    const curLon = ((((ref.lon_deg[0] + 180) % 360) + 360) % 360) - 180;
    const refMarker = L.circleMarker([curLat, curLon], {
      radius: 8,
      color: '#ffffff',
      fillColor: '#ffe66d',
      fillOpacity: 1.0,
      weight: 2.5,
      className: 'current-satellite-marker',
    }).addTo(map);

    refMarker.bindPopup(buildCurrentPopupContent(0));
    layersRef.current.refMarker = refMarker;

    // Current Position Marker (Prediction — Cyan)
    if (pred && pred.lat_deg && pred.lat_deg.length > 0) {
      const predCurLat = pred.lat_deg[0];
      const predCurLon = ((((pred.lon_deg[0] + 180) % 360) + 360) % 360) - 180;
      const predMarker = L.circleMarker([predCurLat, predCurLon], {
        radius: 6,
        color: '#ffffff',
        fillColor: '#00d4ff',
        fillOpacity: 0.9,
        weight: 1.5,
      }).addTo(map);
      layersRef.current.predMarker = predMarker;
    }

    // E. Ground Stations Layer (Requirement 12)
    const stationGroup = L.layerGroup();
    if (result.ground_stations && result.ground_stations.length > 0) {
      result.ground_stations.forEach(gs => {
        const gsMarker = L.circleMarker([gs.lat_deg, gs.lon_deg], {
          radius: 5,
          color: '#4ecdc4',
          fillColor: '#0d1220',
          fillOpacity: 0.9,
          weight: 2,
        });

        const popupContent = `
          <div style="font-family: Inter, sans-serif; font-size: 11px;">
            <div style="font-weight: 600; color: #4ecdc4; font-size: 12px; margin-bottom: 4px; border-bottom: 1px solid #1e2d4a; padding-bottom: 2px;">
              📡 ${gs.name}
            </div>
            <div style="color: #94a3b8; margin-bottom: 2px;">
              <span style="color: #64748b;">Latitude:</span> ${gs.lat_deg.toFixed(4)}°
            </div>
            <div style="color: #94a3b8; margin-bottom: 2px;">
              <span style="color: #64748b;">Longitude:</span> ${gs.lon_deg.toFixed(4)}°
            </div>
            <div style="color: #94a3b8;">
              <span style="color: #64748b;">Elevation:</span> ${gs.height_m.toFixed(0)} m
            </div>
          </div>
        `;

        gsMarker.bindTooltip(`Station: ${gs.name}`, { direction: 'top' });
        gsMarker.bindPopup(popupContent);
        gsMarker.addTo(stationGroup);
      });
    }

    if (showStations) {
      stationGroup.addTo(map);
    }
    layersRef.current.stationGroup = stationGroup;

    // F. Fit bounds smartly (Requirement 10)
    // Avoid over-zooming on small GEO oscillation tracks!
    try {
      const allLats = [...ref.lat_deg];
      const allLons = [...ref.lon_deg];
      if (pred?.lat_deg) {
        allLats.push(...pred.lat_deg);
        allLons.push(...pred.lon_deg);
      }

      const minLat = Math.min(...allLats);
      const maxLat = Math.max(...allLats);
      const minLon = Math.min(...allLons);
      const maxLon = Math.max(...allLons);

      const latSpan = maxLat - minLat;
      const lonSpan = maxLon - minLon;

      // For GEO satellites, track is tight (~1° to 4°).
      // If span is small, ensure a regional view context (e.g. at least 25° lat/lon span)
      let boundsToFit: L.LatLngBounds;
      if (latSpan < 15 && lonSpan < 25) {
        const midLat = (minLat + maxLat) / 2;
        let midLon = (minLon + maxLon) / 2;
        midLon = ((((midLon + 180) % 360) + 360) % 360) - 180;
        boundsToFit = L.latLngBounds(
          [Math.max(-85, midLat - 15), Math.max(-180, midLon - 25)],
          [Math.min(85, midLat + 15), Math.min(180, midLon + 25)]
        );
      } else {
        boundsToFit = refLine.getBounds();
      }

      map.fitBounds(boundsToFit, {
        padding: [30, 30],
        maxZoom: 5, // Prevents excessive zoom in oceans!
        animate: true,
      });
    } catch (e) {
      console.warn('[GroundTrack] fitBounds error:', e);
    }

    map.invalidateSize();
  }, [result, showStations, buildCurrentPopupContent]);

  // ── 5. Update Current Marker when Timeline Index Changes ────────────────
  useEffect(() => {
    if (!result) return;
    const ref = result.reference;
    const pred = result.prediction;
    if (!ref.lat_deg || ref.lat_deg.length === 0) return;

    const safeIdx = Math.max(0, Math.min(currentIndex, ref.lat_deg.length - 1));
    const lat = ref.lat_deg[safeIdx];
    let lon = ref.lon_deg[safeIdx];
    lon = ((((lon + 180) % 360) + 360) % 360) - 180;

    if (layersRef.current.refMarker) {
      layersRef.current.refMarker.setLatLng([lat, lon]);
      // Update popup content live if open
      if (layersRef.current.refMarker.isPopupOpen()) {
        layersRef.current.refMarker.setPopupContent(buildCurrentPopupContent(safeIdx));
      }
    }

    if (pred && pred.lat_deg && layersRef.current.predMarker) {
      const pSafeIdx = Math.max(0, Math.min(currentIndex, pred.lat_deg.length - 1));
      const pLat = pred.lat_deg[pSafeIdx];
      let pLon = pred.lon_deg[pSafeIdx];
      pLon = ((((pLon + 180) % 360) + 360) % 360) - 180;
      layersRef.current.predMarker.setLatLng([pLat, pLon]);
    }
  }, [currentIndex, result, buildCurrentPopupContent]);

  // ── Toggle Ground Station visibility ───────────────────────────────────
  const handleToggleStations = () => {
    const map = mapRef.current;
    const group = layersRef.current.stationGroup;
    const nextState = !showStations;
    setShowStations(nextState);

    if (map && group) {
      if (nextState) {
        if (!map.hasLayer(group)) group.addTo(map);
      } else {
        if (map.hasLayer(group)) map.removeLayer(group);
      }
    }
  };

  // ── Reset Zoom / View Handler ──────────────────────────────────────────
  const handleResetView = () => {
    const map = mapRef.current;
    const refLine = layersRef.current.refLine;
    if (!map || !result) return;

    if (refLine) {
      map.fitBounds(refLine.getBounds(), {
        padding: [30, 30],
        maxZoom: 5,
        animate: true,
      });
    } else {
      map.setView([10, 20], 2);
    }
  };

  // Current telemetry values for bottom bar (Requirement 9 & 16)
  const currentTelemetry = useMemo(() => {
    if (!result) return null;
    const ref = result.reference;
    const idx = Math.max(0, Math.min(currentIndex, (ref.lat_deg?.length || 1) - 1));
    const lat = ref.lat_deg?.[idx];
    const lon = ref.lon_deg?.[idx];
    const alt = ref.alt_km?.[idx];
    const ts = result.timestamps?.[idx] || '';
    const utcStr = ts ? ts.replace('T', ' ').replace(/\.\d+$/, '') + ' UTC' : '—';

    return {
      latStr: lat != null ? `${lat.toFixed(4)}°` : '—',
      lonStr: lon != null ? `${lon.toFixed(4)}°` : '—',
      altStr: alt != null ? `${alt.toFixed(2)} km` : '—',
      utcStr,
    };
  }, [result, currentIndex]);

  return (
    <div className="panel h-full flex flex-col overflow-hidden relative">
      {/* ── Top Header & Controls ──────────────────────────────────────── */}
      <div className="px-3 py-2 border-b border-panel-border flex items-center justify-between bg-panel-bg/95 z-10 flex-wrap gap-2">
        <div className="flex items-center gap-2">
          <h3 className="text-xs font-semibold uppercase tracking-widest text-text-muted">
            Ground Track
          </h3>
          <span className="text-[10px] text-text-muted font-mono bg-space-black px-1.5 py-0.5 rounded border border-panel-border">
            WGS-84
          </span>
        </div>

        {/* Legend (Requirement 6) */}
        <div className="flex items-center gap-3 text-[11px] font-mono select-none">
          <span className="flex items-center gap-1.5 text-text-secondary" title="SGP4 Reference Ground Track">
            <span className="inline-block w-3.5 h-1 bg-[#ffe66d] rounded-sm" />
            <span>REFERENCE — SGP4</span>
          </span>

          {result?.prediction && (
            <span className="flex items-center gap-1.5 text-text-secondary" title="PI-STN Neural Prediction Track">
              <span className="inline-block w-3.5 h-1 border-t-2 border-dashed border-[#00d4ff]" />
              <span>PREDICTION — PI-STN</span>
            </span>
          )}

          <span className="flex items-center gap-1.5 text-text-secondary" title="Current Satellite Position">
            <span className="inline-block w-2.5 h-2.5 rounded-full bg-[#ffe66d] border border-white" />
            <span>Current</span>
          </span>
        </div>

        {/* Action Controls: Tile Switcher & Station Toggle */}
        <div className="flex items-center gap-1.5">
          {/* Tile Selector */}
          <select
            value={activeTile}
            onChange={e => setActiveTile(e.target.value as TileProviderKey)}
            className="bg-space-black border border-panel-border text-[11px] rounded px-1.5 py-0.5 text-text-secondary focus:outline-none focus:border-accent-cyan cursor-pointer"
            title="Base Map Style"
          >
            <option value="dark">Dark Theme</option>
            <option value="osm">Street Map</option>
            <option value="satellite">Satellite</option>
          </select>

          {/* Ground Station Toggle */}
          <button
            onClick={handleToggleStations}
            className={`px-2 py-0.5 rounded text-[11px] border transition-colors ${
              showStations
                ? 'bg-accent-green/20 text-accent-green border-accent-green/40'
                : 'bg-space-black text-text-muted border-panel-border hover:text-text-secondary'
            }`}
            title="Toggle Ground Stations"
          >
            ◆ Stations
          </button>

          {/* Reset View Button */}
          {result && (
            <button
              onClick={handleResetView}
              className="px-2 py-0.5 rounded text-[11px] border border-panel-border text-text-muted hover:text-accent-cyan hover:border-accent-cyan bg-space-black transition-colors"
              title="Fit Track to View"
            >
              ⛶ Fit
            </button>
          )}
        </div>
      </div>

      {/* ── Main Map Canvas (Always Mounted for Immediate Leaflet Render) ─ */}
      <div className="flex-1 min-h-0 relative w-full h-full">
        <div
          ref={containerRef}
          className="w-full h-full"
          style={{ minHeight: '180px' }}
        />

        {/* Empty State Overlay (Requirement 13) */}
        {!result && (
          <div className="absolute inset-0 z-[1000] bg-space-black/75 backdrop-blur-[2px] flex items-center justify-center text-text-muted text-sm pointer-events-auto">
            <div className="text-center p-4 rounded-lg bg-panel-bg/90 border border-panel-border shadow-xl">
              <div className="text-3xl mb-1.5 text-accent-cyan animate-pulse">🛰</div>
              <p className="font-medium text-text-primary text-sm">Ground Track View</p>
              <p className="text-xs text-text-muted mt-1">Run a simulation to plot orbital trajectories</p>
            </div>
          </div>
        )}
      </div>

      {/* ── Compact Information Bar (Requirement 9 & 16) ───────────────── */}
      <div className="px-3 py-1.5 border-t border-panel-border bg-space-black/90 flex items-center justify-between text-xs font-mono select-none z-10">
        <div className="flex items-center gap-4 text-[11px] flex-wrap">
          <div className="flex items-center gap-1.5">
            <span className="text-text-muted">LAT</span>
            <span className="text-accent-gold font-medium">{currentTelemetry?.latStr ?? '—'}</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-text-muted">LON</span>
            <span className="text-accent-gold font-medium">{currentTelemetry?.lonStr ?? '—'}</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-text-muted">ALT</span>
            <span className="text-accent-gold font-medium">{currentTelemetry?.altStr ?? '—'}</span>
          </div>
        </div>

        <div className="text-[11px] text-text-secondary flex items-center gap-1.5">
          <span className="text-text-muted">UTC</span>
          <span className="text-text-primary">{currentTelemetry?.utcStr ?? '—'}</span>
        </div>
      </div>
    </div>
  );
};
