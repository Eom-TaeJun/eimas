#!/bin/bash
# EIMAS 전체 실행 스크립트

set -e

echo "================================================"
echo "  EIMAS - Starting All Services"
echo "================================================"
echo ""

# Create logs directory if not exists
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p logs

# Check if backend is already running
if lsof -i :8000 > /dev/null 2>&1; then
    echo "[WARN] Backend already running on port 8000"
    echo "  To stop: pkill -f 'uvicorn api.main:app'"
else
    echo "[1/2] Starting Backend (FastAPI) on port 8000..."
    uvicorn api.main:app --reload --port 8000 > logs/backend.log 2>&1 &
    BACKEND_PID=$!
    echo "  Backend PID: $BACKEND_PID"
fi

# Wait for backend to be ready
sleep 3

# Check if frontend is already running
if lsof -i :3002 > /dev/null 2>&1; then
    echo "[WARN] Frontend already running on port 3002"
    echo "  To stop: pkill -f 'next dev --port 3002'"
else
    echo "[2/2] Starting Frontend (Next.js) on port 3002..."
    cd frontend

    # Create .env.local if not exists
    if [ ! -f .env.local ]; then
        echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
    fi

    npm run dev -- --port 3002 > ../logs/frontend.log 2>&1 &
    FRONTEND_PID=$!
    echo "  Frontend PID: $FRONTEND_PID"
fi

echo ""
echo "================================================"
echo "  EIMAS is running!"
echo "================================================"
echo ""
echo "Backend:  http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"
echo "Frontend: http://localhost:3002"
echo ""
echo "Logs:"
echo "  Backend:  tail -f logs/backend.log"
echo "  Frontend: tail -f logs/frontend.log"
echo ""
echo "To stop all services:"
echo "  ./stop_all.sh"
echo ""
