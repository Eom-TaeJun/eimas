// Safe number formatting utilities for EIMAS dashboard
// Handles undefined/null values gracefully

/**
 * Safe toFixed that handles undefined/null values
 */
export function safeToFixed(value: number | undefined | null, digits: number = 2): string {
    if (value === undefined || value === null || isNaN(value)) {
        return "0"
    }
    return value.toFixed(digits)
}

/**
 * Safe number formatting with default fallback
 */
export function safeNumber(value: number | undefined | null, fallback: number = 0): number {
    if (value === undefined || value === null || isNaN(value)) {
        return fallback
    }
    return value
}

/**
 * Format money/currency values safely
 * @param value - number in dollars
 * @returns formatted string like "$1.23B" or "$456.7M"
 */
export function formatMoney(value: number | undefined | null): string {
    if (value === undefined || value === null || isNaN(value)) {
        return "$0"
    }
    if (Math.abs(value) >= 1e12) return `$${(value / 1e12).toFixed(2)}T`
    if (Math.abs(value) >= 1e9) return `$${(value / 1e9).toFixed(2)}B`
    if (Math.abs(value) >= 1e6) return `$${(value / 1e6).toFixed(1)}M`
    if (Math.abs(value) >= 1e3) return `$${(value / 1e3).toFixed(1)}K`
    return `$${value.toFixed(0)}`
}

/**
 * Format billions (for FRED data)
 * @param value - number in billions
 * @returns formatted string like "$5.71T" or "$869.3B"
 */
export function formatBillions(value: number | undefined | null): string {
    if (value === undefined || value === null || isNaN(value)) {
        return "$0B"
    }
    if (value >= 1000) {
        return `$${(value / 1000).toFixed(2)}T`
    }
    return `$${value.toFixed(1)}B`
}

/**
 * Format percentage safely
 * @param value - decimal value (0.75 = 75%)
 * @param digits - decimal places
 * @returns formatted string like "75.0%"
 */
export function formatPercent(value: number | undefined | null, digits: number = 1): string {
    if (value === undefined || value === null || isNaN(value)) {
        return "0%"
    }
    return `${(value * 100).toFixed(digits)}%`
}

/**
 * Format basis points from decimal
 * @param value - decimal value (0.0071 = 71bp)
 * @returns formatted string like "+71bp" or "-25bp"
 */
export function formatBasisPoints(value: number | undefined | null): string {
    if (value === undefined || value === null || isNaN(value)) {
        return "0bp"
    }
    const bp = value * 100
    const sign = bp > 0 ? "+" : ""
    return `${sign}${bp.toFixed(0)}bp`
}
