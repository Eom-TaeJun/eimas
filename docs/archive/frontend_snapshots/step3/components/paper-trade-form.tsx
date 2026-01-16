"use client"

import { useState } from "react"
import { useForm } from "react-hook-form"
import { mutate } from "swr"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form"
import { useToast } from "@/hooks/use-toast"
import { Loader2 } from "lucide-react"

interface TradeFormData {
  ticker: string
  side: "BUY" | "SELL"
  quantity: number
}

export function PaperTradeForm() {
  const [isSubmitting, setIsSubmitting] = useState(false)
  const { toast } = useToast()

  const form = useForm<TradeFormData>({
    defaultValues: {
      ticker: "",
      side: "BUY",
      quantity: 1,
    },
  })

  const onSubmit = async (data: TradeFormData) => {
    setIsSubmitting(true)

    try {
      const response = await fetch("/api/paper-trade", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          ticker: data.ticker.toUpperCase(),
          side: data.side,
          quantity: data.quantity,
        }),
      })

      const result = await response.json()

      if (!response.ok) {
        throw new Error(result.error || "Failed to execute trade")
      }

      toast({
        title: "Trade executed successfully",
        description: `${data.side} ${data.quantity} shares of ${data.ticker.toUpperCase()}`,
      })

      // Refresh portfolio and trade history data
      mutate("/api/portfolio")
      mutate((key) => typeof key === "string" && key.startsWith("/api/portfolio/trades"))

      // Reset form
      form.reset()
    } catch (error) {
      toast({
        title: "Trade failed",
        description: error instanceof Error ? error.message : "An error occurred",
        variant: "destructive",
      })
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Paper Trade</CardTitle>
      </CardHeader>
      <CardContent>
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
            <FormField
              control={form.control}
              name="ticker"
              rules={{
                required: "Ticker is required",
                pattern: {
                  value: /^[A-Z]+$/,
                  message: "Ticker must be uppercase letters only",
                },
              }}
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Ticker</FormLabel>
                  <FormControl>
                    <Input
                      placeholder="SPY"
                      {...field}
                      onChange={(e) => field.onChange(e.target.value.toUpperCase())}
                      disabled={isSubmitting}
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="side"
              rules={{ required: "Side is required" }}
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Side</FormLabel>
                  <Select onValueChange={field.onChange} defaultValue={field.value} disabled={isSubmitting}>
                    <FormControl>
                      <SelectTrigger>
                        <SelectValue placeholder="Select side" />
                      </SelectTrigger>
                    </FormControl>
                    <SelectContent>
                      <SelectItem value="BUY">BUY</SelectItem>
                      <SelectItem value="SELL">SELL</SelectItem>
                    </SelectContent>
                  </Select>
                  <FormMessage />
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="quantity"
              rules={{
                required: "Quantity is required",
                min: { value: 1, message: "Quantity must be at least 1" },
              }}
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Quantity</FormLabel>
                  <FormControl>
                    <Input
                      type="number"
                      min="1"
                      {...field}
                      onChange={(e) => field.onChange(Number.parseInt(e.target.value))}
                      disabled={isSubmitting}
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />

            <Button type="submit" className="w-full" disabled={isSubmitting}>
              {isSubmitting && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              Execute Trade
            </Button>
          </form>
        </Form>
      </CardContent>
    </Card>
  )
}
