#!/bin/bash

# EIMAS Dashboard Quick Start Script
# 이 스크립트는 FastAPI 서버와 Next.js 개발 서버를 동시에 실행합니다.

echo "🚀 EIMAS Real-Time Dashboard Starting..."
echo ""

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
    echo ""
fi

# Start FastAPI server in background
echo "🔧 Starting FastAPI server (port 8000)..."
cd ..
uvicorn api.main:app --reload --port 8000 &
FASTAPI_PID=$!
cd dashboard

# Wait for FastAPI to start
sleep 3

# Start Next.js dev server
echo "🎨 Starting Next.js dev server (port 3000)..."
npm run dev &
NEXTJS_PID=$!

echo ""
echo "✅ Dashboard started!"
echo "   FastAPI: http://localhost:8000"
echo "   Dashboard: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop both servers"

# Trap Ctrl+C to kill both processes
trap "kill $FASTAPI_PID $NEXTJS_PID; exit" INT

# Wait for user interrupt
wait
