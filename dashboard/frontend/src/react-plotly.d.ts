declare module 'react-plotly.js' {
  import * as React from 'react';
  import * as Plotly from 'plotly.js';

  export interface PlotParams {
    data: Plotly.Data[];
    layout: Partial<Plotly.Layout>;
    config?: Partial<Plotly.Config>;
    frames?: Plotly.Frame[];
    style?: React.CSSProperties;
    className?: string;
    useResizeHandler?: boolean;
    onInitialized?: (figure: any, graphDiv: HTMLElement) => void;
    onUpdate?: (figure: any, graphDiv: HTMLElement) => void;
    onPurge?: (figure: any, graphDiv: HTMLElement) => void;
    onError?: (err: Error) => void;
  }

  const Plot: React.ComponentType<PlotParams>;
  export default Plot;
}
