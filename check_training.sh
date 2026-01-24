#!/bin/bash

# Check Training Status Script

echo "=========================================="
echo "TRAINING STATUS CHECK"
echo "=========================================="

# Check if pipeline is running
echo ""
echo "1. Checking if pipeline is running..."
if pgrep -f "run_pipeline.py" > /dev/null; then
    echo "   ✓ Pipeline IS running"
    echo "   Process ID(s):"
    pgrep -f "run_pipeline.py" | while read pid; do
        echo "     - PID: $pid"
    done
else
    echo "   ✗ Pipeline is NOT running"
fi

# Check CPU usage
echo ""
echo "2. Python process CPU usage:"
ps aux | grep python | grep -v grep | head -5

# Check for output files
echo ""
echo "3. Recent model outputs:"
if [ -d "outputs" ]; then
    ls -lht outputs/*.pth 2>/dev/null | head -5 || echo "   No .pth files found yet"
else
    echo "   No outputs directory"
fi

# Check training history
echo ""
echo "4. Recent training plots:"
if [ -d "outputs" ]; then
    ls -lht outputs/*.png 2>/dev/null | head -3 || echo "   No .png files found yet"
else
    echo "   No outputs directory"
fi

# Current config
echo ""
echo "5. Current settings:"
if [ -f "config/settings.py" ]; then
    echo "   ENSEMBLE_SIZE: $(grep 'ENSEMBLE_SIZE.*=' config/settings.py | head -1)"
    echo "   NUM_EPOCHS: $(grep 'NUM_EPOCHS.*=' config/settings.py | head -1)"
    echo "   PATIENCE: $(grep 'PATIENCE.*=' config/settings.py | head -1)"
    echo "   HIDDEN_SIZE: $(grep 'HIDDEN_SIZE.*=' config/settings.py | head -1)"
else
    echo "   settings.py not found"
fi

echo ""
echo "=========================================="
echo "QUICK ACTIONS:"
echo "=========================================="
echo "To switch to FAST mode (15-25 min):"
echo "  cp config/settings.py config/settings_full.py"
echo "  cp config/settings_fast.py config/settings.py"
echo ""
echo "To restore FULL mode (40-60 min):"
echo "  cp config/settings_full.py config/settings.py"
echo ""
echo "To kill training (if stuck):"
echo "  pkill -f run_pipeline.py"
echo "=========================================="
