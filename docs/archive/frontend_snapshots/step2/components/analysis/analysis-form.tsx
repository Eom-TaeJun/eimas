"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Checkbox } from "@/components/ui/checkbox"
import { Loader2 } from "lucide-react"
import type { AnalysisLevel, ResearchGoal } from "@/lib/types/analysis"

interface AnalysisFormProps {
  onSubmit: (data: {
    question: string
    analysis_level: string
    research_goal: string
    use_mock: boolean
  }) => Promise<void>
  isLoading: boolean
}

export function AnalysisForm({ onSubmit, isLoading }: AnalysisFormProps) {
  const [question, setQuestion] = useState("")
  const [analysisLevel, setAnalysisLevel] = useState<AnalysisLevel>("Monetary")
  const [researchGoal, setResearchGoal] = useState<ResearchGoal>("Forecasting")
  const [useMock, setUseMock] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!question.trim()) return

    await onSubmit({
      question,
      analysis_level: analysisLevel,
      research_goal: researchGoal,
      use_mock: useMock,
    })
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div className="space-y-2">
        <Label htmlFor="question">Research Question</Label>
        <Textarea
          id="question"
          placeholder="What is the market outlook for the next quarter?"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          disabled={isLoading}
          className="min-h-[120px]"
          required
        />
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <div className="space-y-2">
          <Label htmlFor="analysis-level">Analysis Level</Label>
          <Select
            value={analysisLevel}
            onValueChange={(value) => setAnalysisLevel(value as AnalysisLevel)}
            disabled={isLoading}
          >
            <SelectTrigger id="analysis-level">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="Geopolitics">Geopolitics</SelectItem>
              <SelectItem value="Monetary">Monetary</SelectItem>
              <SelectItem value="Sector">Sector</SelectItem>
              <SelectItem value="Individual">Individual</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <Label htmlFor="research-goal">Research Goal</Label>
          <Select
            value={researchGoal}
            onValueChange={(value) => setResearchGoal(value as ResearchGoal)}
            disabled={isLoading}
          >
            <SelectTrigger id="research-goal">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="Variable Selection">Variable Selection</SelectItem>
              <SelectItem value="Forecasting">Forecasting</SelectItem>
              <SelectItem value="Causal Inference">Causal Inference</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="flex items-center space-x-2">
        <Checkbox
          id="use-mock"
          checked={useMock}
          onCheckedChange={(checked) => setUseMock(checked as boolean)}
          disabled={isLoading}
        />
        <Label htmlFor="use-mock" className="text-sm font-normal cursor-pointer">
          Use mock data for testing
        </Label>
      </div>

      <Button type="submit" disabled={isLoading || !question.trim()} className="w-full">
        {isLoading ? (
          <>
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            Running Analysis...
          </>
        ) : (
          "Run Analysis"
        )}
      </Button>
    </form>
  )
}
