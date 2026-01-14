import Link from "next/link"
import { RealTimeClock } from "./RealTimeClock"

export function Navbar() {
  return (
    <nav className="border-b border-[#30363d] bg-[#0d1117]">
      <div className="mx-auto max-w-[1400px] px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 items-center justify-between">
          <div className="flex items-center gap-8">
            <Link href="/" className="text-xl font-bold text-white">
              EIMAS
            </Link>
            <div className="hidden md:flex items-center gap-6">
              <Link href="/" className="text-sm text-gray-300 hover:text-white transition-colors">
                Dashboard
              </Link>
              <Link href="/analysis" className="text-sm text-gray-400 hover:text-white transition-colors">
                Analysis
              </Link>
              <Link href="/portfolio" className="text-sm text-gray-400 hover:text-white transition-colors">
                Portfolio
              </Link>
            </div>
          </div>
          <RealTimeClock />
        </div>
      </div>
    </nav>
  )
}
