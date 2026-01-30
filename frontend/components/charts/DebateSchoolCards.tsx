"use client"

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Brain, MessageSquare, AlertCircle, CheckCircle2 } from "lucide-react"
import type { EIMASAnalysis } from "@/lib/types"

interface DebateSchoolCardsProps {
    data: EIMASAnalysis["debate_consensus"]["enhanced"]
}

export function DebateSchoolCards({ data }: DebateSchoolCardsProps) {
    if (!data?.interpretation?.school_interpretations) return null

    const schools = data.interpretation.school_interpretations

    const getStanceColor = (stance: string) => {
        switch (stance) {
            case "BULLISH": return "bg-green-500/10 text-green-400 border-green-500/20"
            case "BEARISH": return "bg-red-500/10 text-red-400 border-red-500/20"
            case "NEUTRAL": return "bg-yellow-500/10 text-yellow-400 border-yellow-500/20"
            default: return "bg-gray-500/10 text-gray-400 border-gray-500/20"
        }
    }

    const getSchoolIcon = (school: string) => {
        if (school.includes("Economist")) return <Brain className="w-5 h-5 text-blue-400" />
        if (school.includes("Devil")) return <AlertCircle className="w-5 h-5 text-red-400" />
        if (school.includes("Risk")) return <CheckCircle2 className="w-5 h-5 text-orange-400" />
        return <MessageSquare className="w-5 h-5 text-gray-400" />
    }

    return (
        <Card className="bg-[#161b22] border-[#30363d]">
            <CardHeader>
                <CardTitle className="text-xl font-bold text-white flex items-center gap-2">
                    <MessageSquare className="w-5 h-5 text-blue-400" />
                    AI Analysts Debate
                </CardTitle>
                <CardDescription className="text-gray-400">
                    Diverse interpretations from different economic schools
                </CardDescription>
            </CardHeader>
            <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {schools.map((school, idx) => (
                        <Card key={idx} className="bg-[#0d1117] border border-[#30363d] overflow-hidden">
                            <div className="p-4 border-b border-[#30363d] flex items-center justify-between bg-[#1c2128]/50">
                                <div className="flex items-center gap-2">
                                    {getSchoolIcon(school.school)}
                                    <span className="font-semibold text-gray-200 text-sm">{school.school}</span>
                                </div>
                                <Badge variant="outline" className={getStanceColor(school.stance)}>
                                    {school.stance}
                                </Badge>
                            </div>
                            <div className="p-4">
                                <div className="h-[150px] pr-4 overflow-y-auto">
                                    <ul className="space-y-3">
                                        {school.reasoning.map((point, i) => (
                                            <li key={i} className="text-sm text-gray-400 leading-relaxed flex items-start gap-2">
                                                <span className="text-[#30363d] mt-1.5">•</span>
                                                <span>{point}</span>
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                            </div>
                        </Card>
                    ))}
                </div>
            </CardContent>
        </Card>
    )
}
