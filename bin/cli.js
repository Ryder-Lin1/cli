#!/usr/bin/env node
const fs = require("fs");
const path = require("path");

const rawArgs = process.argv.slice(2);

let command = "create";
let projectName;

if (rawArgs[0] === "create" || rawArgs[0] === "install") {
  command = rawArgs[0];
  projectName = rawArgs[1];
} else {
  projectName = rawArgs[0];
}

if (!projectName) {
  projectName = "my-chesshacks-bot";
}

if (command === "install") {
  console.log("Install command is not implemented yet.");
  process.exit(0);
}

const projectPath = path.join(process.cwd(), projectName);
const starterPath = path.join(__dirname, "starter");

fs.mkdirSync(projectPath, { recursive: true });

function copyRecursive(src, dest) {
  const stats = fs.statSync(src);

  if (stats.isDirectory()) {
    fs.mkdirSync(dest, { recursive: true });
    const entries = fs.readdirSync(src);
    for (const entry of entries) {
      const srcEntry = path.join(src, entry);
      const destEntry = path.join(dest, entry);
      copyRecursive(srcEntry, destEntry);
    }
  } else {
    const destDir = path.dirname(dest);
    fs.mkdirSync(destDir, { recursive: true });
    const contents = fs.readFileSync(src);
    fs.writeFileSync(dest, contents);
  }
}

copyRecursive(starterPath, projectPath);

console.log(`✅ Created ${projectName}`);
