#!/bin/bash
# Setup script: Copy VTES model to OBS plugin data directory
# Run this after building the OBS plugin

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VTES_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL_SRC="$VTES_ROOT/yolo_training/runs/vtes_obb_10k-2/weights/best.onnx"
MODEL_JSON="$VTES_ROOT/yolo_training/runs/vtes_obb_10k-2/weights/best.json"

# OBS plugin data directory (Linux default)
OBS_PLUGIN_DATA="${OBS_PLUGIN_DATA:-$HOME/.config/obs-studio/plugins/vtes_obs_detect/data}"

if [ ! -f "$MODEL_SRC" ]; then
    echo "ERROR: Model not found at $MODEL_SRC"
    echo "Run training first: cd yolo_training && python3 train_obb_10k.py"
    exit 1
fi

echo "Setting up VTES OBS Plugin..."
echo ""

# Create plugin data directory
mkdir -p "$OBS_PLUGIN_DATA/models"

# Copy model files
cp "$MODEL_SRC" "$OBS_PLUGIN_DATA/models/vtes_card_1080.onnx"
cp "$MODEL_JSON" "$OBS_PLUGIN_DATA/models/vtes_card_1080.json"

echo "Model copied to: $OBS_PLUGIN_DATA/models/"
echo ""
echo "OBS Setup Instructions:"
echo "1. Build and install the vtes_obs_detect plugin"
echo "2. Add the filter to your webcam source in OBS"
echo "3. In filter settings:"
echo "   - Enable 'Advanced'"
echo "   - Set Model Size to 'External Model'"
echo "   - Set Model Path to: $OBS_PLUGIN_DATA/models/vtes_card_1080.onnx"
echo "   - Enable 'VTES Card Detection'"
echo "   - Set WebSocket Host: 127.0.0.1"
echo "   - Set WebSocket Port: 3998"
echo "   - Set Cooldown: 10 seconds"
echo ""
echo "4. Start the Node.js server:"
echo "   cd $VTES_ROOT/vtes-obs-plugin && npm start"
echo ""
echo "5. Add a Browser Source in OBS with URL: http://localhost:3999"
