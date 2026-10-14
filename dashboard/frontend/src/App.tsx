// src/App.tsx
import React from 'react';
import { Header } from './components/layout/Header';
import { Dashboard } from './pages/Dashboard';

const App: React.FC = () => {
  return (
    <div className="flex flex-col h-screen overflow-hidden bg-space-black">
      <Header />
      <div className="flex-1 overflow-hidden">
        <Dashboard />
      </div>
    </div>
  );
};

export default App;
