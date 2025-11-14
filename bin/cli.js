#!/usr/bin/env node
const fs = require("fs");
const path = require("path");

const projectName = process.argv[2] || "my-chesshacks-bot";
const projectPath = path.join(process.cwd(), projectName);

fs.mkdirSync(projectPath, { recursive: true });

fs.writeFileSync(
  path.join(projectPath, "main.py"),
  'def main():\n    print("Hello, World!")\n\nif __name__ == "__main__":\n    main()\n'
);

fs.writeFileSync(path.join(projectPath, "requirements.txt"), "");

fs.writeFileSync(
  path.join(projectPath, "README.md"),
  `# ${projectName}\n\nA ChessHacks bot.\n`
);

console.log(`✅ Created ${projectName}`);
