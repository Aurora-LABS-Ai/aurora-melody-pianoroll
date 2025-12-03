# Aurora Melody Piano Roll

A professional MIDI piano roll editor with AI-powered melody generation capabilities. Available as standalone application and VST3 plugin.

## Installation

### Standalone Application

1. Run `Aurora-Melody.exe` directly
2. The `assets` folder must remain in the same directory as the executable

### VST3 Plugin

1. Copy `Aurora-Melody.vst3` folder to your VST3 plugins directory:
   - Windows: `C:\Program Files\Common Files\VST3\`
2. Rescan plugins in your DAW

## Features

- Professional piano roll editor with FL Studio-style workflow
- MIDI import/export support
- Built-in synthesizer with SoundFont support
- AI melody generation via plugin system
- Velocity editing lane
- Snap-to-grid with multiple resolution options
- Undo/Redo support

## Included Plugins

The `plugins` folder contains generator plugins:

| Plugin | Description |
|--------|-------------|
| `random-melody.aml` | Random melody generator with scale selection |
| `arpeggiator.aml` | Chord arpeggiator with pattern options |
| `chord-progression.aml` | Chord progression builder |
| `ostinato-generator.aml` | Ostinato pattern generator |

## AI Service (Optional)

The `ai-service` folder contains a Python server for AI-powered melody generation.

### Setup

```bash
cd ai-service
pip install -r requirements.txt
```

### Configuration

1. Copy `.env.example` to `.env`
2. Add your API key (e.g., Google Gemini API key)
3. Edit `config.yaml` for additional settings

### Running the Server

Windows:
```bash
start_server.bat
```

Linux/macOS:
```bash
./start_server.sh
```

The server runs on `http://localhost:8765` by default.

## Creating Custom Plugins

Use the Aurora Melody SDK to create your own generator plugins:

https://github.com/Aurora-LABS-Ai/aurora-melody-sdk

### Quick Start

```bash
pip install aurora-melody-sdk
```

```python
from aurora_melody_sdk import GeneratorPlugin, MidiNote, Scale

class MyPlugin(GeneratorPlugin):
    name = "My Generator"
    
    def generate(self, context):
        notes = []
        # Your generation logic here
        return notes
```

Package your plugin:
```bash
aurora-pack ./my-plugin
```

Place the resulting `.aml` file in the `plugins` folder.

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Space | Play/Stop (resets to beginning) |
| Ctrl+Z | Undo |
| Ctrl+Y | Redo |
| Ctrl+A | Select All |
| Delete | Delete selected notes |
| Ctrl+C | Copy |
| Ctrl+V | Paste |
| Ctrl+X | Cut |

## System Requirements

- Windows 10 or later (64-bit)
- 4GB RAM minimum
- VST3 compatible DAW (for plugin version)

## License

MIT License

## Links

- SDK Repository: https://github.com/Aurora-LABS-Ai/aurora-melody-sdk
- Issue Tracker: https://github.com/Aurora-LABS-Ai/aurora-melody-pianoroll/issues

