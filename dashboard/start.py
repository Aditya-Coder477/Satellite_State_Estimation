#!/usr/bin/env python3
"""
dashboard/start.py
==================
One-command startup for the Satellite Orbit Lab.

Usage (from project root):
    python dashboard/start.py

Starts:
  - FastAPI backend on port 8000 (localhost)
  - Vite frontend dev server on port 5173 (localhost)

Then open: http://localhost:5173
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DASHBOARD = ROOT / "dashboard"
FRONTEND = DASHBOARD / "frontend"
BACKEND = DASHBOARD / "backend"


def main():
    print("=" * 60)
    print("  SATELLITE ORBIT LAB — Starting")
    print("=" * 60)
    print(f"  Project root : {ROOT}")
    print(f"  Backend      : http://localhost:8000")
    print(f"  Frontend     : http://localhost:5173")
    print("=" * 60)

    # Check frontend deps
    node_modules = FRONTEND / "node_modules"
    if not node_modules.exists():
        print("\n[setup] Installing frontend dependencies...")
        subprocess.run(["npm", "install"], cwd=str(FRONTEND), check=True, shell=True)
        print("[setup] Frontend deps installed.")

    # Determine Python executable (prefer project venv if available)
    venv_py = ROOT / "venv" / "Scripts" / "python.exe"
    py_exe = str(venv_py) if venv_py.is_file() else sys.executable

    # Start backend
    print(f"\n[backend] Starting FastAPI with {py_exe}...")
    backend_proc = subprocess.Popen(
        [
            py_exe, "-m", "uvicorn",
            "dashboard.backend.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload",
        ],
        cwd=str(ROOT),
    )

    time.sleep(2)

    # Start frontend
    print("\n[frontend] Starting Vite dev server...")
    frontend_proc = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=str(FRONTEND),
        shell=True,
    )

    print("\n✓ Satellite Orbit Lab is running!")
    print("  Open: http://localhost:5173\n")
    print("  Press Ctrl+C to stop.\n")

    try:
        backend_proc.wait()
        frontend_proc.wait()
    except KeyboardInterrupt:
        print("\n[stop] Shutting down...")
        backend_proc.terminate()
        frontend_proc.terminate()


if __name__ == "__main__":
    main()
