# chesshacks

A CLI tool for creating a starter bot for ChessHacks.

## Usage

```bash
npx chesshacks [create/install]
```

### create

```bash
npx chesshacks create <bot-name>
```

Creates a new bot in the current directory, with scaffolded starter code and devtools. Keep in mind that devtools are gitignored, so people who clone your repo will need to run `npx chesshacks install` to install devtools.

### install

```bash
npx chesshacks install
```

Installs ChessHacks devtools in the current directory. These devtools will be gitignored.
