"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { PortfolioPie } from "@/components/charts/PortfolioPie"

export function PortfolioComposition() {
  const [composition, setComposition] = useState<Record<string, number>>({})
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function fetchData() {
      try {
        const res = await fetch("/api/risk")
        const json = await res.json()
        setComposition(json.composition || {})
      } catch (error) {
        console.error("[v0] Failed to fetch portfolio composition:", error)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [])

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Portfolio Composition</CardTitle>
          <CardDescription>Loading...</CardDescription>
        </CardHeader>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Portfolio Composition</CardTitle>
        <CardDescription>Asset allocation by ticker</CardDescription>
      </CardHeader>
      <CardContent>
        <PortfolioPie weights={composition} />
      </CardContent>
    </Card>
  )
}
