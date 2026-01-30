#!/bin/bash
# Merge all frontend steps into single project

set -e

echo "🔀 Merging frontend steps..."

# Create target directory
rm -rf frontend
mkdir -p frontend

# Merge files from all steps
for step in frontend_steps/step*; do
  echo "   Merging $step..."
  rsync -av --ignore-existing "$step/" frontend/
done

# Check result
echo ""
echo "✅ Merged frontend structure:"
tree -L 2 frontend/ 2>/dev/null || find frontend/ -type f | head -20

echo ""
echo "📦 Next steps:"
echo "   cd frontend"
echo "   npm install"
echo "   npm run dev"
