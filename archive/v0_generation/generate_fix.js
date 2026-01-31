#!/usr/bin/env node
const fs = require('fs');
const path = require('path');

const apiKey = process.env.V0_API_KEY;

async function generateFix() {
  const prompt = fs.readFileSync('prompts/fix_nextjs_config.md', 'utf-8');
  
  console.log('Generating Next.js config fix...');
  console.log(`Prompt: ${prompt.length} chars\n`);

  try {
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
        name: 'Next.js Config Fix',
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
    console.log(`Chat: ${chat.id}`);
    console.log(`URL: ${chat.webUrl}\n`);

    if (chat.latestVersion?.files) {
      console.log('Generated files:');
      for (const file of chat.latestVersion.files) {
        console.log(`  ${file.name}`);
        console.log(file.content);
        console.log('---');
      }
    }

    console.log(`\nView: ${chat.webUrl}`);

  } catch (error) {
    console.error('Error:', error.message);
    process.exit(1);
  }
}

generateFix();
