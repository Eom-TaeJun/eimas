#!/usr/bin/env node
/**
 * Generate EIMAS step-by-step
 */

const fs = require('fs');
const path = require('path');

const stepNumber = process.argv[2];
if (!stepNumber) {
  console.error('Usage: node generate_step.js <step_number>');
  process.exit(1);
}

const promptFile = `prompts/step${stepNumber}_*.md`;
const apiKey = process.env.V0_API_KEY;

async function generateStep() {
  const files = fs.readdirSync('prompts').filter(f => f.startsWith(`step${stepNumber}_`));

  if (files.length === 0) {
    console.error(`❌ No prompt found for step ${stepNumber}`);
    process.exit(1);
  }

  const promptPath = path.join('prompts', files[0]);
  const prompt = fs.readFileSync(promptPath, 'utf-8');

  console.log(`🚀 Step ${stepNumber}: ${files[0]}`);
  console.log(`📝 Prompt: ${prompt.length} chars\n`);

  try {
    console.log('🎨 Creating v0 chat (2-3 min)...');

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 300000);

    const response = await fetch('https://api.v0.dev/v1/chats', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
        'User-Agent': 'eimas/1.0'
      },
      body: JSON.stringify({
        name: `EIMAS Step ${stepNumber}`,
        privacy: 'private',
        message: prompt
      }),
      signal: controller.signal
    });

    clearTimeout(timeout);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${await response.text()}`);
    }

    const chat = await response.json();
    console.log(`✅ Chat: ${chat.id}`);
    console.log(`   URL: ${chat.webUrl}\n`);

    // Wait for completion
    if (chat.latestVersion?.status === 'completed') {
      console.log('✅ Generation completed immediately!');
    } else {
      console.log('⏳ Processing... Check URL for progress');
    }

    // Save result
    const resultDir = `frontend_steps/step${stepNumber}`;
    fs.mkdirSync(resultDir, { recursive: true });

    fs.writeFileSync(
      path.join(resultDir, 'generation.json'),
      JSON.stringify({
        step: stepNumber,
        chatId: chat.id,
        webUrl: chat.webUrl,
        demoUrl: chat.latestVersion?.demoUrl,
        status: chat.latestVersion?.status,
        generatedAt: new Date().toISOString()
      }, null, 2)
    );

    if (chat.latestVersion?.files) {
      for (const file of chat.latestVersion.files) {
        const filePath = path.join(resultDir, file.name);
        fs.mkdirSync(path.dirname(filePath), { recursive: true });
        fs.writeFileSync(filePath, file.content);
        console.log(`   ✓ ${file.name}`);
      }
    }

    console.log(`\n🎉 Done! View: ${chat.webUrl}`);

  } catch (error) {
    if (error.name === 'AbortError') {
      console.error('❌ Timeout after 5 minutes');
    } else {
      console.error('❌ Error:', error.message);
    }
    process.exit(1);
  }
}

generateStep();
