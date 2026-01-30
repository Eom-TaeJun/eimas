#!/usr/bin/env node
/**
 * EIMAS Web App Generator using v0 SDK
 *
 * This script uses the v0 Platform API to generate a complete web application
 * for the EIMAS system based on the prompt in v0_prompt.md
 */

const fs = require('fs');
const path = require('path');
const { V0 } = require('v0-sdk');

// Load V0 API key from environment
const V0_API_KEY = process.env.V0_API_KEY;

if (!V0_API_KEY) {
  console.error('Error: V0_API_KEY environment variable not set');
  console.error('Please set your V0 API key:');
  console.error('  export V0_API_KEY=your-key-here');
  process.exit(1);
}

// Initialize v0 client
const v0 = new V0({
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

    // Create a new v0 project
    console.log('🎨 Creating v0 project...');
    const project = await v0.projects.create({
      name: 'eimas-web-app',
      description: 'EIMAS Economic Intelligence Multi-Agent System Web Application',
      framework: 'nextjs', // Next.js 14+ with App Router
      prompt: prompt
    });

    console.log(`✅ Project created: ${project.id}`);
    console.log(`   Project URL: https://v0.dev/projects/${project.id}\n`);

    // Wait for project generation to complete
    console.log('⏳ Waiting for code generation...');
    console.log('   This may take 1-2 minutes...\n');

    let status = 'processing';
    let generation;
    let attempts = 0;
    const maxAttempts = 12; // 1 minute max (5 second intervals)

    while (status === 'processing' && attempts < maxAttempts) {
      await new Promise(resolve => setTimeout(resolve, 5000)); // Wait 5 seconds

      generation = await v0.projects.getGeneration(project.id);
      status = generation.status;
      attempts++;

      console.log(`   Status: ${status} (attempt ${attempts}/${maxAttempts})`);
    }

    if (status === 'completed') {
      console.log('\n✅ Code generation completed!\n');

      // Get the generated files
      const files = await v0.projects.getFiles(project.id);

      console.log(`📦 Generated ${files.length} files:\n`);

      // Create output directory
      const outputDir = path.join(__dirname, 'frontend');
      if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
      }

      // Save all generated files
      for (const file of files) {
        const filePath = path.join(outputDir, file.path);
        const fileDir = path.dirname(filePath);

        // Create directory if it doesn't exist
        if (!fs.existsSync(fileDir)) {
          fs.mkdirSync(fileDir, { recursive: true });
        }

        // Write file
        fs.writeFileSync(filePath, file.content, 'utf-8');
        console.log(`   ✓ ${file.path}`);
      }

      console.log(`\n✅ All files saved to ./frontend/\n`);

      // Generate a summary file
      const summary = {
        project_id: project.id,
        project_url: `https://v0.dev/projects/${project.id}`,
        generated_at: new Date().toISOString(),
        files_count: files.length,
        files: files.map(f => f.path)
      };

      fs.writeFileSync(
        path.join(outputDir, 'generation_summary.json'),
        JSON.stringify(summary, null, 2)
      );

      console.log('📊 Generation summary:');
      console.log(`   Project ID: ${project.id}`);
      console.log(`   Files generated: ${files.length}`);
      console.log(`   Output directory: ./frontend/`);
      console.log(`   View project: https://v0.dev/projects/${project.id}\n`);

      console.log('🎉 Done! Next steps:');
      console.log('   1. cd frontend');
      console.log('   2. npm install');
      console.log('   3. Create .env.local file with API_URL');
      console.log('   4. npm run dev\n');

    } else if (status === 'failed') {
      console.error('❌ Code generation failed');
      console.error(`   Error: ${generation.error || 'Unknown error'}`);
      process.exit(1);
    } else {
      console.error('❌ Code generation timed out after 1 minute');
      console.error(`   Last status: ${status}`);
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
