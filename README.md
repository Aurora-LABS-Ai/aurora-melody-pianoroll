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

## AI Melody Generation

Aurora Melody uses a custom-trained AI model for melody generation. The model is available on Hugging Face.

### Step 1: Download the Model

Download all files from the Hugging Face repository:

https://huggingface.co/alvanalrakib/aurora-melody

Required files:
- `best_model.safetensors` - Model weights
- `best_model_config.json` - Model configuration
- `vocabulary.pkl` - Tokenizer vocabulary
- `aurora_tokenizer-*.whl` - Aurora Tokenizer wheel file

### Step 2: Install Aurora Tokenizer

Install the tokenizer wheel file from the downloaded files:

```bash
pip install aurora_tokenizer-<version>.whl
```

### Step 3: Install AI Service Dependencies

```bash
cd ai-service
pip install -r requirements.txt
```

### Step 4: Configure Model Paths

Edit `ai-service/config.yaml` and update the paths to your downloaded model files:

```yaml
model:
  # Path to model weights (.safetensors)
  weights: "C:/path/to/your/models/best_model.safetensors"

  # Path to model config (.json)
  config: "C:/path/to/your/models/best_model_config.json"
  
  # Path to vocabulary (.pkl)
  vocab: "C:/path/to/your/models/vocabulary.pkl"
```

Use forward slashes (/) or double backslashes (\\) for Windows paths.

### Step 5: Run the AI Server

Windows:
```bash
cd ai-service
start_server.bat
```

Linux/macOS:
```bash
cd ai-service
./start_server.sh
```

The server runs on `http://localhost:8765` by default.

### Step 6: Use AI Plugin

1. Open Aurora Melody
2. Go to the Plugins tab
3. Select an AI plugin (e.g., Gemini Melody AI)
4. Configure generation parameters
5. Click Generate

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
- Python 3.8+ (for AI service)
- CUDA compatible GPU (optional, for faster AI generation)
- VST3 compatible DAW (for plugin version)

## License

MIT License

## Links

- Model: https://huggingface.co/alvanalrakib/aurora-melody
- SDK: https://github.com/Aurora-LABS-Ai/aurora-melody-sdk
- Issues: https://github.com/Aurora-LABS-Ai/aurora-melody-pianoroll/issues

