"use client"

import { useState, useEffect } from "react"
import { useTheme } from "next-themes"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Checkbox } from "@/components/ui/checkbox"
import { Switch } from "@/components/ui/switch"
import { Button } from "@/components/ui/button"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { useToast } from "@/hooks/use-toast"

interface EIMASSettings {
  accountName: string
  initialCashBalance: string
  defaultAnalysisLevel: string
  defaultResearchGoal: string
  skipDataStage: boolean
  skipAnalysisStage: boolean
  skipDebateStage: boolean
  autoRefreshInterval: string
  autoRefreshEnabled: boolean
}

const defaultSettings: EIMASSettings = {
  accountName: "default",
  initialCashBalance: "100000",
  defaultAnalysisLevel: "standard",
  defaultResearchGoal: "balanced",
  skipDataStage: false,
  skipAnalysisStage: false,
  skipDebateStage: false,
  autoRefreshInterval: "60",
  autoRefreshEnabled: true,
}

export default function SettingsPage() {
  const [settings, setSettings] = useState<EIMASSettings>(defaultSettings)
  const [mounted, setMounted] = useState(false)
  const { theme, setTheme } = useTheme()
  const { toast } = useToast()

  // Load settings from localStorage on mount
  useEffect(() => {
    setMounted(true)
    const savedSettings = localStorage.getItem("eimas-settings")
    if (savedSettings) {
      try {
        setSettings(JSON.parse(savedSettings))
      } catch (error) {
        console.error("[v0] Failed to parse settings:", error)
      }
    }
  }, [])

  const handleSave = () => {
    localStorage.setItem("eimas-settings", JSON.stringify(settings))
    toast({
      title: "Settings Saved",
      description: "Your preferences have been saved successfully.",
    })
  }

  const updateSetting = <K extends keyof EIMASSettings>(key: K, value: EIMASSettings[K]) => {
    setSettings((prev) => ({ ...prev, [key]: value }))
  }

  // Prevent hydration mismatch for theme toggle
  if (!mounted) {
    return null
  }

  return (
    <div className="container mx-auto p-6 max-w-4xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-balance">EIMAS Settings</h1>
        <p className="text-muted-foreground mt-2">Configure your investment analysis system preferences</p>
      </div>

      <div className="space-y-6">
        {/* Account Settings */}
        <Card>
          <CardHeader>
            <CardTitle>Account Settings</CardTitle>
            <CardDescription>Configure your account name and initial balance</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="accountName">Account Name</Label>
              <Input
                id="accountName"
                value={settings.accountName}
                onChange={(e) => updateSetting("accountName", e.target.value)}
                placeholder="default"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="cashBalance">Initial Cash Balance ($)</Label>
              <Input
                id="cashBalance"
                type="number"
                value={settings.initialCashBalance}
                onChange={(e) => updateSetting("initialCashBalance", e.target.value)}
                placeholder="100000"
              />
            </div>
          </CardContent>
        </Card>

        {/* Analysis Settings */}
        <Card>
          <CardHeader>
            <CardTitle>Analysis Settings</CardTitle>
            <CardDescription>Set default analysis parameters and stage preferences</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="analysisLevel">Default Analysis Level</Label>
              <Select
                value={settings.defaultAnalysisLevel}
                onValueChange={(value) => updateSetting("defaultAnalysisLevel", value)}
              >
                <SelectTrigger id="analysisLevel">
                  <SelectValue placeholder="Select level" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="basic">Basic</SelectItem>
                  <SelectItem value="standard">Standard</SelectItem>
                  <SelectItem value="advanced">Advanced</SelectItem>
                  <SelectItem value="comprehensive">Comprehensive</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="researchGoal">Default Research Goal</Label>
              <Select
                value={settings.defaultResearchGoal}
                onValueChange={(value) => updateSetting("defaultResearchGoal", value)}
              >
                <SelectTrigger id="researchGoal">
                  <SelectValue placeholder="Select goal" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="growth">Growth</SelectItem>
                  <SelectItem value="income">Income</SelectItem>
                  <SelectItem value="balanced">Balanced</SelectItem>
                  <SelectItem value="value">Value</SelectItem>
                  <SelectItem value="aggressive">Aggressive</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-3 pt-2">
              <Label>Skip Analysis Stages</Label>
              <div className="space-y-3">
                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="skipData"
                    checked={settings.skipDataStage}
                    onCheckedChange={(checked) => updateSetting("skipDataStage", checked === true)}
                  />
                  <label
                    htmlFor="skipData"
                    className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 cursor-pointer"
                  >
                    Skip Data Collection Stage
                  </label>
                </div>
                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="skipAnalysis"
                    checked={settings.skipAnalysisStage}
                    onCheckedChange={(checked) => updateSetting("skipAnalysisStage", checked === true)}
                  />
                  <label
                    htmlFor="skipAnalysis"
                    className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 cursor-pointer"
                  >
                    Skip Analysis Stage
                  </label>
                </div>
                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="skipDebate"
                    checked={settings.skipDebateStage}
                    onCheckedChange={(checked) => updateSetting("skipDebateStage", checked === true)}
                  />
                  <label
                    htmlFor="skipDebate"
                    className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 cursor-pointer"
                  >
                    Skip Debate Stage
                  </label>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Data Refresh Settings */}
        <Card>
          <CardHeader>
            <CardTitle>Data Refresh Settings</CardTitle>
            <CardDescription>Configure automatic data refresh behavior</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label htmlFor="autoRefresh">Enable Auto-Refresh</Label>
                <p className="text-sm text-muted-foreground">Automatically refresh data at set intervals</p>
              </div>
              <Switch
                id="autoRefresh"
                checked={settings.autoRefreshEnabled}
                onCheckedChange={(checked) => updateSetting("autoRefreshEnabled", checked)}
              />
            </div>

            <div className="space-y-3">
              <Label>Refresh Interval</Label>
              <RadioGroup
                value={settings.autoRefreshInterval}
                onValueChange={(value) => updateSetting("autoRefreshInterval", value)}
                disabled={!settings.autoRefreshEnabled}
              >
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="30" id="interval30" />
                  <Label htmlFor="interval30" className="font-normal cursor-pointer">
                    30 seconds
                  </Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="60" id="interval60" />
                  <Label htmlFor="interval60" className="font-normal cursor-pointer">
                    60 seconds
                  </Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="120" id="interval120" />
                  <Label htmlFor="interval120" className="font-normal cursor-pointer">
                    120 seconds
                  </Label>
                </div>
              </RadioGroup>
            </div>
          </CardContent>
        </Card>

        {/* Theme Settings */}
        <Card>
          <CardHeader>
            <CardTitle>Theme Settings</CardTitle>
            <CardDescription>Customize the appearance of your interface</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label htmlFor="theme">Dark Mode</Label>
                <p className="text-sm text-muted-foreground">Toggle between light and dark theme</p>
              </div>
              <Switch
                id="theme"
                checked={theme === "dark"}
                onCheckedChange={(checked) => setTheme(checked ? "dark" : "light")}
              />
            </div>
          </CardContent>
        </Card>

        {/* Save Button */}
        <div className="flex justify-end">
          <Button onClick={handleSave} size="lg">
            Save Settings
          </Button>
        </div>
      </div>
    </div>
  )
}
