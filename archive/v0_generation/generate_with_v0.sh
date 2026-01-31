#!/bin/bash

# EIMAS Web App Generator using v0 API (curl version)
# This script uses curl to call v0 API directly

set -e

echo "🚀 Starting EIMAS web app generation with v0..."
echo ""

# Check for API key
if [ -z "$V0_API_KEY" ]; then
    echo "❌ Error: V0_API_KEY environment variable not set"
    echo "Please set your V0 API key:"
    echo "  export V0_API_KEY=your-key-here"
    exit 1
fi

# Read the prompt
PROMPT_FILE="v0_prompt.md"
if [ ! -f "$PROMPT_FILE" ]; then
    echo "❌ Error: $PROMPT_FILE not found"
    exit 1
fi

PROMPT=$(cat "$PROMPT_FILE" | jq -Rs .)
echo "📝 Loaded prompt from v0_prompt.md"
echo "   Prompt length: $(cat "$PROMPT_FILE" | wc -c) characters"
echo ""

# Create the chat
echo "🎨 Creating v0 chat..."

CHAT_RESPONSE=$(curl -s -X POST https://api.v0.dev/v1/chats \
  -H "Authorization: Bearer $V0_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{
    \"name\": \"EIMAS Web Application\",
    \"privacy\": \"private\",
    \"message\": $PROMPT
  }")

# Check for errors
if echo "$CHAT_RESPONSE" | jq -e '.error' > /dev/null 2>&1; then
    echo "❌ Error creating chat:"
    echo "$CHAT_RESPONSE" | jq -r '.error'
    exit 1
fi

CHAT_ID=$(echo "$CHAT_RESPONSE" | jq -r '.id')
CHAT_URL=$(echo "$CHAT_RESPONSE" | jq -r '.webUrl')

echo "✅ Chat created: $CHAT_ID"
echo "   Chat URL: $CHAT_URL"
echo ""

# Wait for generation to complete
echo "⏳ Waiting for code generation..."
echo "   This may take 1-2 minutes..."
echo ""

MAX_ATTEMPTS=12  # 1 minute (5 second intervals)
ATTEMPT=0
STATUS="pending"

while [ "$STATUS" = "pending" ] && [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    sleep 5
    ATTEMPT=$((ATTEMPT + 1))

    CHAT_DATA=$(curl -s -X GET "https://api.v0.dev/v1/chats/$CHAT_ID" \
      -H "Authorization: Bearer $V0_API_KEY")

    if echo "$CHAT_DATA" | jq -e '.latestVersion.status' > /dev/null 2>&1; then
        STATUS=$(echo "$CHAT_DATA" | jq -r '.latestVersion.status')
        echo "   Status: $STATUS (attempt $ATTEMPT/$MAX_ATTEMPTS)"
    else
        echo "   Status: pending (attempt $ATTEMPT/$MAX_ATTEMPTS)"
    fi
done

if [ "$STATUS" = "completed" ]; then
    echo ""
    echo "✅ Code generation completed!"
    echo ""

    # Get the files
    FILES=$(echo "$CHAT_DATA" | jq -r '.latestVersion.files')
    FILE_COUNT=$(echo "$FILES" | jq '. | length')

    echo "📦 Generated $FILE_COUNT files:"
    echo ""

    # Create output directory
    OUTPUT_DIR="frontend"
    mkdir -p "$OUTPUT_DIR"

    # Save all files
    for i in $(seq 0 $((FILE_COUNT - 1))); do
        FILE_NAME=$(echo "$FILES" | jq -r ".[$i].name")
        FILE_CONTENT=$(echo "$FILES" | jq -r ".[$i].content")
        FILE_PATH="$OUTPUT_DIR/$FILE_NAME"

        # Create directory if needed
        mkdir -p "$(dirname "$FILE_PATH")"

        # Write file
        echo "$FILE_CONTENT" > "$FILE_PATH"
        echo "   ✓ $FILE_NAME"
    done

    echo ""
    echo "✅ All files saved to ./frontend/"
    echo ""

    # Save summary
    DEMO_URL=$(echo "$CHAT_DATA" | jq -r '.latestVersion.demoUrl // empty')
    SCREENSHOT_URL=$(echo "$CHAT_DATA" | jq -r '.latestVersion.screenshotUrl // empty')

    cat > "$OUTPUT_DIR/generation_summary.json" <<EOF
{
  "chat_id": "$CHAT_ID",
  "chat_url": "$CHAT_URL",
  "demo_url": "$DEMO_URL",
  "screenshot_url": "$SCREENSHOT_URL",
  "generated_at": "$(date -Iseconds)",
  "files_count": $FILE_COUNT,
  "files": $(echo "$FILES" | jq '[.[].name]')
}
EOF

    echo "📊 Generation summary:"
    echo "   Chat ID: $CHAT_ID"
    echo "   Files generated: $FILE_COUNT"
    echo "   Output directory: ./frontend/"
    echo "   View project: $CHAT_URL"
    [ -n "$DEMO_URL" ] && echo "   Live demo: $DEMO_URL"
    echo ""

    echo "🎉 Done! Next steps:"
    echo "   1. cd frontend"
    echo "   2. npm install"
    echo "   3. Create .env.local file with:"
    echo "      NEXT_PUBLIC_API_URL=http://localhost:8000"
    echo "   4. npm run dev"
    echo ""

elif [ "$STATUS" = "failed" ]; then
    echo ""
    echo "❌ Code generation failed"
    exit 1
else
    echo ""
    echo "❌ Code generation timed out"
    echo "   Last status: $STATUS"
    echo "   You can still view the chat at: $CHAT_URL"
    exit 1
fi
