/** @type {import('next').NextConfig} */
const nextConfig = {
  // Fix workspace root detection
  outputFileTracingRoot: __dirname,
}

module.exports = nextConfig
