"use client"

import { useState, useEffect } from "react"
import { format } from "date-fns"

export function RealTimeClock() {
  const [currentTime, setCurrentTime] = useState(new Date())

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date())
    }, 1000)

    return () => clearInterval(timer)
  }, [])

  return <div className="text-sm text-gray-400 font-mono">{format(currentTime, "MMM dd, yyyy HH:mm:ss")}</div>
}
