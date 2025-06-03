# English Agent Fine-tuning with Unsloth - Complete Tutorial

This notebook demonstrates how to fine-tune an LLM using Unsloth specifically for the English Agent in your ADK Multi-Agent Educational system.

## 🎯 Objectives

1. Create a specialized dataset for English language education
2. Fine-tune a language model using Unsloth for 2x faster training
3. Integrate the fine-tuned model back into your ADK system
4. Evaluate performance improvements

## 📚 Prerequisites

1. **GPU Requirements**: NVIDIA GPU with CUDA support (recommended: T4, RTX series, A100)
2. **Memory**: At least 8GB VRAM (16GB+ recommended)
3. **Python Environment**: Python 3.10-3.12

## 🚀 Installation

First, install the fine-tuning requirements:

```bash
pip install -r finetuning/requirements-finetuning.txt
```

**Note**: The exact unsloth installation command depends on your CUDA version and PyTorch version. Run this to get the optimal command:

```bash
wget -qO- https://raw.githubusercontent.com/unslothai/unsloth/main/unsloth/_auto_install.py | python -
```

## 💾 1. Dataset Preparation

### Educational English Dataset Structure

Our dataset focuses on three main areas:
- **Grammar Correction & Explanation**
- **Writing Improvement & Style**
- **Vocabulary & Word Usage**

### Sample Dataset Format

```python
{
    "messages": [
        {
            "role": "system",
            "content": "You are a specialized English Language Education Agent..."
        },
        {
            "role": "user", 
            "content": "Please fix the grammar in this sentence: 'Me and my friend went to store.'"
        },
        {
            "role": "assistant",
            "content": "**Corrected sentence:** My friend and I went to the store.\n\n**Grammar explanation:**..."
        }
    ]
}
```

## 🔧 2. Fine-tuning Code

### Model Setup

```python
from unsloth import FastLanguageModel
import torch

# Choose model based on your needs
MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"  # Fast, educational use
# MODEL_NAME = "unsloth/gemma-2-9b-bnb-4bit"  # More capable
# MODEL_NAME = "unsloth/Phi-3.5-mini-instruct"  # Compact

MAX_SEQ_LENGTH = 2048
```

### Training Configuration

```python
# LoRA configuration for efficient training
LORA_CONFIG = {
    "r": 16,  # Rank - higher = more parameters but better quality
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    "lora_alpha": 16,
    "lora_dropout": 0,  # Optimized for Unsloth
    "bias": "none",
    "use_gradient_checkpointing": "unsloth"
}

# Training parameters
TRAINING_CONFIG = {
    "per_device_train_batch_size": 1,  # Adjust for your GPU
    "gradient_accumulation_steps": 4,   # Effective batch size = 4
    "warmup_steps": 10,
    "max_steps": 100,  # Increase for more training
    "learning_rate": 2e-4,
    "logging_steps": 1,
    "optim": "adamw_8bit",
    "output_dir": "./english_agent_finetuned"
}
```

## 📊 3. Educational Benefits

### Why Fine-tune for English Education?

1. **Specialized Vocabulary**: Educational terms, grammar terminology
2. **Teaching Style**: Step-by-step explanations, examples, memory tricks
3. **Student-Appropriate Language**: Age-appropriate explanations
4. **Curriculum Alignment**: Common grammar issues, writing patterns

### Expected Improvements

- **Better Grammar Explanations**: More structured, rule-based responses
- **Consistent Teaching Style**: Follows educational best practices
- **Domain-Specific Knowledge**: Better understanding of English education context
- **Reduced Hallucinations**: More accurate grammar rules and examples

## 🔄 4. Integration with ADK

After fine-tuning, you can integrate the model in several ways:

### Option 1: Local Model Integration
```python
# Update english_agent/__init__.py
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./english_agent_finetuned/final_model",
    max_seq_length=2048,
    load_in_4bit=True
)

# Create custom LLM wrapper for ADK
class FineTunedEnglishAgent:
    def __init__(self):
        self.model = model
        self.tokenizer = tokenizer
    
    def generate_response(self, prompt):
        # Implementation for ADK integration
        pass
```

### Option 2: GGUF Export for Ollama
```python
# Export to GGUF for local deployment
model.save_pretrained_gguf(
    "english_agent_model", 
    tokenizer,
    quantization_method="q4_k_m"
)
```

### Option 3: Hugging Face Hub Upload
```python
# Upload to HF Hub for cloud deployment
model.push_to_hub("your-username/english-education-agent")
tokenizer.push_to_hub("your-username/english-education-agent")
```

## 📈 5. Evaluation & Testing

### Test Cases for English Agent

```python
test_cases = [
    "Fix this grammar: 'The group of students are studying hard.'",
    "Explain the difference between affect and effect",
    "Help me write a thesis statement about climate change",
    "What's wrong with: 'Between you and I, this is difficult'",
    "Explain passive vs active voice with examples"
]
```

### Performance Metrics

- **Grammar Accuracy**: Correct identification and fixing of errors
- **Explanation Quality**: Clear, educational explanations
- **Teaching Consistency**: Follows educational principles
- **Response Speed**: Faster inference with optimized model

## 🎓 6. Advanced Techniques

### Dataset Augmentation
- Use existing grammar datasets (JFLEG, Lang8)
- Generate synthetic examples using larger models
- Include real student writing samples (anonymized)

### Multi-task Learning
- Combine grammar correction, style improvement, and vocabulary
- Include reading comprehension tasks
- Add creative writing assistance

### Curriculum-Based Training
- Start with basic grammar, progress to complex writing
- Include grade-level appropriate examples
- Align with common educational standards

## 📁 File Structure

After fine-tuning, your project structure will look like:

```
adk-multi-agent-educator/
├── finetuning/
│   ├── requirements-finetuning.txt
│   ├── prepare_dataset.py
│   ├── finetune_english_agent.py
│   ├── evaluate_model.py
│   └── english_agent_finetuned/
│       ├── final_model/
│       ├── checkpoints/
│       └── logs/
└── src/adk_educator/sub_agents/english_agent/
    ├── __init__.py (updated with fine-tuned model)
    └── fine_tuned_model/ (optional local copy)
```

## 🚀 Next Steps

1. **Run the complete fine-tuning script**
2. **Evaluate the model with test cases**
3. **Integrate with your ADK system**
4. **Compare performance with the original model**
5. **Collect user feedback and iterate**

## 💡 Tips for Success

1. **Start Small**: Use a smaller dataset first to validate the pipeline
2. **Monitor Training**: Watch for overfitting, adjust learning rate
3. **Iterative Improvement**: Fine-tune based on real usage patterns
4. **Educational Validation**: Have teachers review the outputs
5. **Performance Tracking**: Monitor response quality over time

## 🔗 Resources

- [Unsloth Documentation](https://docs.unsloth.ai)
- [Educational Dataset Sources](https://huggingface.co/datasets?search=grammar)
- [Grammar Correction Benchmarks](https://github.com/grammarly/gector)
- [Educational AI Best Practices](https://www.edweek.org/technology/ai-in-education)

---

Ready to create your specialized English education AI agent! 🎓✨
