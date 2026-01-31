#!/usr/bin/env node
/**
 * EIMAS Web App Generator using v0 SDK
 *
 * This script uses the v0 Platform API to generate a complete web application
 * for the EIMAS system based on the prompt in v0_prompt.md
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { createClient } from 'v0-sdk';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load V0 API key from environment
const V0_API_KEY = process.env.V0_API_KEY;

if (!V0_API_KEY) {
  console.error('Error: V0_API_KEY environment variable not set');
  console.error('Please set your V0 API key:');
  console.error('  export V0_API_KEY=your-key-here');
  process.exit(1);
}

// Initialize v0 client
const v0 = createClient({
  apiKey: V0_API_KEY
});

async function main() {
  try {
    console.log('🚀 Starting EIMAS web app generation with v0...\n');

    // Read the prompt from v0_prompt.md
    const promptPath = path.join(__dirname, 'v0_prompt.md');
    const prompt = fs.readFileSync(promptPath, 'utf-8');

    console.log('📝 Loaded prompt from v0_prompt.md');
    console.log(`   Prompt length: ${prompt.length} characters\n`);

    // Create a new chat with the prompt
    console.log('🎨 Creating v0 chat...');
    const chat = await v0.chats.create({
      name: 'EIMAS Web Application',
      privacy: 'private',
      message: prompt
    });

    console.log(`✅ Chat created: ${chat.id}`);
    console.log(`   Chat URL: ${chat.webUrl}\n`);

    // Wait for generation to complete
    console.log('⏳ Waiting for code generation...');
    console.log('   This may take 1-2 minutes...\n');

    let attempts = 0;
    const maxAttempts = 12; // 1 minute max (5 second intervals)
    let latestVersion = chat.latestVersion;

    while ((!latestVersion || latestVersion.status === 'pending') && attempts < maxAttempts) {
      await new Promise(resolve => setTimeout(resolve, 5000)); // Wait 5 seconds

      const updatedChat = await v0.chats.getById({ id: chat.id });
      latestVersion = updatedChat.latestVersion;
      attempts++;

      console.log(`   Status: ${latestVersion?.status || 'pending'} (attempt ${attempts}/${maxAttempts})`);
    }

    if (latestVersion?.status === 'completed') {
      console.log('\n✅ Code generation completed!\n');

      const files = latestVersion.files || [];
      console.log(`📦 Generated ${files.length} files:\n`);

      // Create output directory
      const outputDir = path.join(__dirname, 'frontend');
      if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
      }

      // Save all generated files
      for (const file of files) {
        const filePath = path.join(outputDir, file.name);
        const fileDir = path.dirname(filePath);

        // Create directory if it doesn't exist
        if (!fs.existsSync(fileDir)) {
          fs.mkdirSync(fileDir, { recursive: true });
        }

        // Write file
        fs.writeFileSync(filePath, file.content, 'utf-8');
        console.log(`   ✓ ${file.name}`);
      }

      console.log(`\n✅ All files saved to ./frontend/\n`);

      // Generate a summary file
      const summary = {
        chat_id: chat.id,
        chat_url: chat.webUrl,
        demo_url: latestVersion.demoUrl,
        screenshot_url: latestVersion.screenshotUrl,
        generated_at: new Date().toISOString(),
        files_count: files.length,
        files: files.map(f => f.name)
      };

      fs.writeFileSync(
        path.join(outputDir, 'generation_summary.json'),
        JSON.stringify(summary, null, 2)
      );

      console.log('📊 Generation summary:');
      console.log(`   Chat ID: ${chat.id}`);
      console.log(`   Files generated: ${files.length}`);
      console.log(`   Output directory: ./frontend/`);
      console.log(`   View project: ${chat.webUrl}`);
      if (latestVersion.demoUrl) {
        console.log(`   Live demo: ${latestVersion.demoUrl}`);
      }
      console.log();

      console.log('🎉 Done! Next steps:');
      console.log('   1. cd frontend');
      console.log('   2. npm install');
      console.log('   3. Create .env.local file with:');
      console.log('      NEXT_PUBLIC_API_URL=http://localhost:8000');
      console.log('   4. npm run dev\n');

    } else if (latestVersion?.status === 'failed') {
      console.error('❌ Code generation failed');
      process.exit(1);
    } else {
      console.error('❌ Code generation timed out');
      console.error(`   Last status: ${latestVersion?.status || 'unknown'}`);
      console.error(`   You can still view the chat at: ${chat.webUrl}`);
      process.exit(1);
    }

  } catch (error) {
    console.error('❌ Error:', error.message);

    if (error.response) {
      console.error('   Status:', error.response.status);
      console.error('   Data:', JSON.stringify(error.response.data, null, 2));
    }

    process.exit(1);
  }
}

// Run the script
main();
