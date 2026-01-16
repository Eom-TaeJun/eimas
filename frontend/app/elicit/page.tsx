"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Loader2, AlertTriangle, CheckCircle, TrendingUp, Activity } from "lucide-react"

export default function ElicitResearchPage() {
  const [query, setQuery] = useState("Analyze current market conditions and identify key risks for next month.")
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)

  const runResearch = async () => {
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await fetch("http://localhost:8000/api/research", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: query,
          quick_mode: true
        }),
      })

      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`)
      }

      const data = await response.json()
      setResult(data)
    } catch (err: any) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container mx-auto p-6 space-y-8">
      <div className="flex flex-col space-y-2">
        <h1 className="text-3xl font-bold tracking-tight">Elicit Deep Research</h1>
        <p className="text-muted-foreground">
          Autonomous multi-agent research for market analysis and fact-checking.
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-1">
        <Card>
          <CardHeader>
            <CardTitle>Research Query</CardTitle>
            <CardDescription>
              Enter your research question or topic for the agent swarm to analyze.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Textarea 
              placeholder="e.g., Analyze the impact of recent Fed comments on tech stocks." 
              className="min-h-[100px]"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
            <Button onClick={runResearch} disabled={loading}>
              {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              {loading ? "Researching..." : "Run Research"}
            </Button>
          </CardContent>
        </Card>

        {error && (
          <Alert variant="destructive">
            <AlertTriangle className="h-4 w-4" />
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {result && (
          <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
            {/* Top Summary Cards */}
            <div className="grid gap-4 md:grid-cols-3">
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Recommendation</CardTitle>
                  <TrendingUp className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{result.recommendation}</div>
                  <p className="text-xs text-muted-foreground">
                    Confidence: {(result.confidence * 100).toFixed(0)}%
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Risk Score</CardTitle>
                  <Activity className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{result.risk_score.toFixed(1)}/100</div>
                  <p className="text-xs text-muted-foreground">
                    Regime: {result.regime?.regime || "Unknown"}
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Consensus</CardTitle>
                  {result.agreement ? (
                    <CheckCircle className="h-4 w-4 text-green-500" />
                  ) : (
                    <AlertTriangle className="h-4 w-4 text-yellow-500" />
                  )}
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{result.agreement ? "Agreed" : "Dissent"}</div>
                  <p className="text-xs text-muted-foreground">
                    Full: {result.full_mode_position} | Ref: {result.reference_mode_position}
                  </p>
                </CardContent>
              </Card>
            </div>

            {/* Devil's Advocate Section */}
            <Card className="border-l-4 border-l-yellow-500">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <AlertTriangle className="h-5 w-5 text-yellow-500" />
                  Devil's Advocate (Counter-Arguments)
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="list-disc pl-5 space-y-2">
                  {result.devil_advocate && result.devil_advocate.length > 0 ? (
                    result.devil_advocate.map((arg: string, index: number) => (
                      <li key={index} className="text-sm">{arg}</li>
                    ))
                  ) : (
                    <li className="text-sm text-muted-foreground">No strong counter-arguments detected.</li>
                  )}
                </ul>
              </CardContent>
            </Card>

             {/* Detailed JSON View (Optional) */}
             <Card>
              <CardHeader>
                <CardTitle>Raw Data</CardTitle>
              </CardHeader>
              <CardContent>
                <pre className="bg-slate-950 text-slate-50 p-4 rounded-lg overflow-x-auto text-xs">
                  {JSON.stringify(result, null, 2)}
                </pre>
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </div>
  )
}
