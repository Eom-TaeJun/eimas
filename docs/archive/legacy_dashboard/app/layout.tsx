import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'EIMAS Real-Time Dashboard',
  description: 'Economic Intelligence Multi-Agent System - Real-Time Dashboard',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
