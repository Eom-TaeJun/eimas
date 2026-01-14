# Fix Next.js Workspace Root Configuration

## Problem
Next.js is detecting multiple lockfiles and inferring wrong workspace root.
Error: "Next.js inferred your workspace root, but it may not be correct"

## Requirements

Create proper Next.js configuration files to fix workspace root detection:

1. **next.config.js**
   - Set outputFileTracingRoot to current directory
   - Compatible with Next.js 16.0.10
   - TypeScript support

2. **.npmrc**
   - Isolate package management to current directory
   - Prevent parent directory lockfile conflicts

3. **package.json adjustments** (if needed)
   - Ensure "private": true
   - Correct project name and version

## Current Setup
- Next.js: 16.0.10
- Directory: /home/tj/projects/autoai/eimas/frontend
- Parent has package-lock.json at /home/tj/package-lock.json
- Multiple lockfiles causing conflict

## Expected Output

Provide complete configuration files that:
- Fix workspace root to current directory
- Prevent lockfile conflicts
- Work with Next.js 16 Turbopack
- No experimental flags needed

Format: Ready-to-use configuration files with comments.
