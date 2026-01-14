#!/bin/bash

echo "================================================"
echo "  EIMAS - Stopping All Services"
echo "================================================"
echo ""

# Stop Backend
echo "[1/2] Stopping Backend (FastAPI)..."
pkill -f "uvicorn api.main:app"
if [ $? -eq 0 ]; then
    echo "  Backend stopped"
else
    echo "  No backend process found"
fi
echo ""

# Stop Frontend
echo "[2/2] Stopping Frontend (Next.js)..."
pkill -f "next dev --port 3002"
if [ $? -eq 0 ]; then
    echo "  Frontend stopped"
else
    echo "  No frontend process found"
fi
echo ""

# Verify
echo "================================================"
echo "  Verification"
echo "================================================"
echo ""

BACKEND_PID=$(pgrep -f "uvicorn api.main:app")
FRONTEND_PID=$(pgrep -f "next dev")

if [ -z "$BACKEND_PID" ] && [ -z "$FRONTEND_PID" ]; then
    echo "All services stopped successfully"
else
    if [ ! -z "$BACKEND_PID" ]; then
        echo "Warning: Backend still running (PID: $BACKEND_PID)"
        echo "  Force kill: kill -9 $BACKEND_PID"
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        echo "Warning: Frontend still running (PID: $FRONTEND_PID)"
        echo "  Force kill: kill -9 $FRONTEND_PID"
    fi
fi
echo ""
