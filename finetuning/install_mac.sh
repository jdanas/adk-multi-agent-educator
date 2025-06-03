#!/bin/bash

echo "🍎 Installing Fine-tuning Dependencies for Mac M4"
echo "================================================="

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "⚠️  This script is designed for macOS. You may need to adjust for your OS."
fi

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "🐍 Python version: $python_version"

if [[ "$python_version" < "3.8" ]]; then
    echo "❌ Python 3.8+ required. Current version: $python_version"
    exit 1
fi

# Upgrade pip
echo "📦 Upgrading pip..."
python3 -m pip install --upgrade pip

# Install PyTorch with MPS support
echo "🔥 Installing PyTorch with MPS support..."
python3 -m pip install torch torchvision torchaudio

# Install other requirements
echo "📚 Installing fine-tuning requirements..."
python3 -m pip install -r finetuning/requirements-finetuning-mac.txt

# Verify MPS availability
echo "🧪 Testing MPS availability..."
python3 -c "
import torch
if torch.backends.mps.is_available():
    print('✅ MPS is available! Apple Silicon acceleration ready.')
    device = torch.device('mps')
    print(f'🔥 MPS device: {device}')
    # Test basic operation
    x = torch.randn(5, 3, device=device)
    print('✅ MPS test successful!')
else:
    print('⚠️  MPS not available. Will use CPU.')
"

echo ""
echo "✅ Installation complete!"
echo ""
echo "🚀 Next steps:"
echo "1. Run: python3 finetuning/finetune_english_agent_mac.py"
echo "2. Wait for training to complete (~10-20 minutes)"
echo "3. Test integration with your ADK system"
echo ""
echo "💡 If you see any errors, check the troubleshooting section in the README."
