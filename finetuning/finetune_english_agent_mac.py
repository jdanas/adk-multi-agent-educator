"""
Fine-tune English Agent for Mac M4 (Apple Silicon)
Uses MPS (Metal Performance Shaders) instead of CUDA for efficient training
"""

import torch
import json
import os
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

# Configuration
MODEL_NAME = "microsoft/DialoGPT-small"  # Mac-friendly model
OUTPUT_DIR = "./english_agent_finetuned"
MAX_LENGTH = 512

def check_device():
    """Check available device (MPS for Mac M-series)"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"✅ Using Apple Silicon MPS acceleration")
    else:
        device = torch.device("cpu")
        print(f"⚠️ MPS not available, using CPU")
    
    print(f"🔥 Device: {device}")
    return device

def load_model_and_tokenizer():
    """Load a Mac-compatible model for educational fine-tuning"""
    print(f"📥 Loading model: {MODEL_NAME}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("🔧 Set pad_token to eos_token")
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,  # Use float16 for efficiency
            device_map="auto",
            trust_remote_code=True
        )
        
        print(f"✅ Model loaded successfully")
        print(f"📊 Model parameters: {model.num_parameters():,}")
        
        # Print model architecture for debugging
        print("🔍 Model architecture (first few layers):")
        for name, module in list(model.named_modules())[:10]:
            if hasattr(module, 'weight'):
                print(f"   {name}: {type(module).__name__}")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise

def create_educational_dataset():
    """Create comprehensive educational English dataset"""
    
    print("📚 Creating educational dataset...")
    
    educational_data = [
        # Grammar Correction Examples
        {
            "input": "Fix this grammar: Me and my friend went to store.",
            "output": "**Corrected:** My friend and I went to the store.\n\n**Grammar Rule:** Use 'I' when it's the subject of the sentence (doing the action). Use 'me' when it's the object (receiving the action).\n\n**Memory Trick:** Remove the other person from the sentence. 'I went to the store' sounds correct, while 'Me went to the store' doesn't.\n\n**Practice:** Try this with other sentences!"
        },
        {
            "input": "What's wrong with: 'The group of students are studying hard.'",
            "output": "**Corrected:** The group of students is studying hard.\n\n**Grammar Rule:** When the subject is a collective noun like 'group', 'team', 'family', use a singular verb.\n\n**Why:** 'Group' is singular, so we use 'is' not 'are'.\n\n**More Examples:**\n• The team is winning.\n• The family is traveling.\n• The class is learning."
        },
        {
            "input": "Fix: 'The book who I read was interesting.'",
            "output": "**Corrected:** The book that I read was interesting.\n\n**Grammar Rule:** Use 'that' or 'which' for things, 'who' for people.\n\n**Remember:**\n• WHO = people (The person who called)\n• THAT/WHICH = things (The book that I read)\n\n**Practice:** The movie _____ we watched was great. (Answer: that)"
        },
        
        # Vocabulary and Word Usage
        {
            "input": "What's the difference between affect and effect?",
            "output": "**AFFECT vs EFFECT:**\n\n• **AFFECT** = Verb (action word)\n  - 'The rain will affect our picnic plans.'\n  - Memory trick: **A**ffect = **A**ction\n\n• **EFFECT** = Noun (thing/result)\n  - 'The effect of rain was a cancelled picnic.'\n  - Memory trick: **E**ffect = **E**nd result\n\n**Quick Test:** Try substituting 'influence' (verb) or 'result' (noun) in your sentence!"
        },
        {
            "input": "Explain the difference between who and whom.",
            "output": "**WHO vs WHOM Made Simple:**\n\n**WHO** = Subject (does the action)\n- 'WHO is calling?' (Someone is calling)\n- 'The person WHO called was helpful.'\n\n**WHOM** = Object (receives the action)\n- 'To WHOM am I speaking?' (I am speaking to someone)\n- 'The person WHOM I called was helpful.'\n\n**Quick Trick:** \n- If you can answer with HE/SHE → use WHO\n- If you can answer with HIM/HER → use WHOM\n\nExample: '(Who/Whom) did you see?'\nAnswer: 'I saw HIM' → Use WHOM!"
        },
        {
            "input": "When do I use 'less' vs 'fewer'?",
            "output": "**LESS vs FEWER:**\n\n**FEWER** = Countable things (you can count them)\n- 'Fewer books' (1 book, 2 books, 3 books...)\n- 'Fewer students'\n- 'Fewer cookies'\n\n**LESS** = Uncountable things (you can't count individual units)\n- 'Less water' (not 1 water, 2 water...)\n- 'Less time'\n- 'Less sugar'\n\n**Memory Trick:** If you can put a number in front of it, use FEWER!"
        },
        
        # Writing Improvement
        {
            "input": "Help me improve this sentence: The thing is really good.",
            "output": "**Original:** The thing is really good.\n\n**Problems:**\n• 'Thing' is too vague\n• 'Really' is overused\n• 'Good' is generic\n\n**Improved Options:**\n1. 'The presentation was exceptionally well-organized.'\n2. 'This novel offers compelling character development.'\n3. 'The restaurant serves outstanding Italian cuisine.'\n\n**Writing Tips:**\n• Replace vague words with specific terms\n• Use stronger adjectives (exceptional, outstanding, remarkable)\n• Be concrete and descriptive"
        },
        {
            "input": "How do I write a strong thesis statement?",
            "output": "**Strong Thesis Statement Formula:**\n\n**Weak:** Social media is bad.\n**Strong:** While social media connects people globally, excessive use among teenagers significantly impacts their mental health, academic performance, and real-world social skills.\n\n**Key Elements:**\n1. **Specific topic** (social media for teenagers)\n2. **Clear position** (acknowledges benefits but argues negative impacts)\n3. **Main points preview** (mental health, academics, social skills)\n4. **Arguable** (someone could disagree)\n\n**Your Turn:** [Topic] + [Your position] + [Main supporting points] = Strong thesis!"
        },
        {
            "input": "What's wrong with this sentence: 'Between you and I, this is difficult.'",
            "output": "**Corrected:** Between you and me, this is difficult.\n\n**Grammar Rule:** After prepositions (between, with, for, to), use object pronouns.\n\n**Object Pronouns:** me, him, her, us, them\n**Subject Pronouns:** I, he, she, we, they\n\n**Examples:**\n• Between you and me (not I)\n• Come with him and me (not he and I)\n• This is for her and me (not she and I)\n\n**Memory Trick:** Remove the other person - 'between I' sounds wrong, 'between me' sounds right!"
        },
        
        # Reading Comprehension and Analysis
        {
            "input": "How do I analyze the main idea of a paragraph?",
            "output": "**Finding Main Ideas - Step by Step:**\n\n**1. Look for the Topic Sentence**\n- Usually the first or last sentence\n- States the main point clearly\n\n**2. Identify Supporting Details**\n- Examples, facts, explanations\n- They all relate back to the main idea\n\n**3. Ask Key Questions:**\n- What is this paragraph mostly about?\n- What point is the author trying to make?\n\n**4. Summarize in One Sentence**\n- If you can't, the paragraph might have multiple ideas\n\n**Practice:** Read a paragraph and ask 'What's the one big idea here?'"
        }
    ]
    
    print(f"📊 Created {len(educational_data)} educational examples")
    return Dataset.from_list(educational_data)

def format_dataset(dataset, tokenizer):
    """Format dataset for training with educational prompt structure"""
    
    def format_prompt(example):
        # Create educational conversation format
        prompt = f"""<|im_start|>system
You are a specialized English Language Education Agent. Your role is to help students with grammar, writing, vocabulary, and language concepts. Always provide clear explanations with examples and teaching principles.<|im_end|>
<|im_start|>user
{example['input']}<|im_end|>
<|im_start|>assistant
{example['output']}<|im_end|>"""
        return {"text": prompt}
    
    print("🔄 Formatting dataset...")
    formatted_dataset = dataset.map(format_prompt)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
    
    print("🔢 Tokenizing dataset...")
    tokenized_dataset = formatted_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=formatted_dataset.column_names
    )
    
    return tokenized_dataset

def find_target_modules(model):
    """Find the correct target modules for LoRA in the model"""
    target_modules = []
    
    print("🔍 Scanning model architecture for LoRA targets...")
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and len(module.weight.shape) == 2:  # Linear layers
            module_name = name.split('.')[-1]
            target_modules.append(module_name)
            print(f"   Found: {name} -> {module_name}")
    
    # Remove duplicates and get unique module names
    unique_modules = list(set(target_modules))
    
    # Prefer attention-related modules if available
    attention_modules = [mod for mod in unique_modules if any(
        pattern in mod.lower() for pattern in ['attn', 'attention', 'query', 'key', 'value', 'proj', 'dense', 'c_attn', 'c_proj']
    )]
    
    # If we found attention modules, use them; otherwise use any linear modules
    if attention_modules:
        final_modules = attention_modules[:4]  # Limit to first 4
        print(f"✅ Using attention modules: {final_modules}")
    else:
        # Use the first few linear modules
        final_modules = unique_modules[:4]
        print(f"⚠️ No attention modules found, using linear modules: {final_modules}")
    
    return final_modules

def setup_lora_config(model):
    """Configure LoRA for efficient fine-tuning with correct target modules"""
    print("⚙️ Setting up LoRA configuration...")
    
    try:
        # Find the correct target modules for this model
        target_modules = find_target_modules(model)
        
        if not target_modules:
            # Fallback: use all linear layers (less efficient but works)
            print("⚠️ Falling back to all linear layers...")
            target_modules = "all-linear"
        
        print(f"🎯 LoRA target modules: {target_modules}")
        
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,  # Reduced rank for more compatibility
            lora_alpha=16,  # Reduced alpha
            lora_dropout=0.1,
            target_modules=target_modules,
            bias="none",
            inference_mode=False,
        )
        
    except Exception as e:
        print(f"❌ Error setting up LoRA config: {e}")
        print("💡 Trying minimal LoRA configuration...")
        
        # Minimal fallback configuration
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=4,  # Very small rank
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules="all-linear",  # Target all linear layers
            bias="none",
            inference_mode=False,
        )

def test_finetuned_model(model, tokenizer):
    """Test the fine-tuned model with sample questions"""
    print("\n🧪 Testing fine-tuned model...")
    
    test_questions = [
        "Fix this grammar: 'Me and Sarah went to the store.'",
        "What's the difference between 'your' and 'you're'?",
        "Help me improve this sentence: 'The thing was really good.'"
    ]
    
    model.eval()
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n🔍 Test {i}: {question}")
        
        # Format input
        prompt = f"""<|im_start|>system
You are a specialized English Language Education Agent.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        
        try:
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
            print(f"🤖 Response: {response[:200]}...")
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")

def main():
    """Main fine-tuning function for Mac M4"""
    print("🍎 Starting Mac M4 Fine-tuning for English Agent")
    print("=" * 60)
    
    # Check device compatibility
    device = check_device()
    
    # Load model and tokenizer
    try:
        model, tokenizer = load_model_and_tokenizer()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Create educational dataset
    dataset = create_educational_dataset()
    tokenized_dataset = format_dataset(dataset, tokenizer)
    
    # Setup LoRA for efficient training
    try:
        lora_config = setup_lora_config(model)
        model = get_peft_model(model, lora_config)
        print("✅ LoRA adapters applied successfully")
        
        # Verify LoRA was applied correctly
        lora_applied = any("lora" in name.lower() for name, _ in model.named_parameters())
        if not lora_applied:
            raise ValueError("LoRA adapters were not applied correctly")
            
    except Exception as e:
        print(f"❌ LoRA setup failed: {e}")
        print("🔄 Trying alternative LoRA configuration...")
        
        # Fallback: simpler LoRA config
        fallback_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules="all-linear",
            bias="none",
        )
        
        try:
            model = get_peft_model(model, fallback_config)
            print("✅ Fallback LoRA configuration applied")
        except Exception as e2:
            print(f"❌ Fallback LoRA also failed: {e2}")
            print("💡 Continuing with full fine-tuning (will use more memory)")
            # Continue without LoRA - will fine-tune all parameters
    
    # Print trainable parameters info
    try:
        trainable_params = model.print_trainable_parameters()
        print(f"📈 Trainable parameters: {trainable_params}")
    except:
        # Alternative way to check trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"📈 Total parameters: {total_params:,}")
        print(f"📈 Trainable parameters: {trainable_params:,}")
        print(f"📈 Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    # Training arguments optimized for Mac M4 (fixed compatibility)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,  # Small batch for Mac memory
        gradient_accumulation_steps=4,  # Effective batch size = 4
        num_train_epochs=3,
        learning_rate=2e-4,
        warmup_steps=10,
        logging_steps=5,
        save_steps=25,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_pin_memory=False,  # Better for Mac
        fp16=True,  # Mixed precision for efficiency
        report_to="none",  # Disable wandb/tensorboard (fixed)
        load_best_model_at_end=False,
        # Removed evaluation_strategy as it's not available in this version
        push_to_hub=False,
    )
    
    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Start training
    print("🚀 Starting training...")
    print(f"📊 Dataset size: {len(tokenized_dataset)} examples")
    print(f"⏱️ Estimated time: 10-20 minutes")
    
    try:
        trainer.train()
        print("✅ Training completed successfully!")
        
        # Save the fine-tuned model
        print("💾 Saving fine-tuned model...")
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        
        # Test the model
        test_finetuned_model(model, tokenizer)
        
        print(f"\n🎉 Fine-tuning complete!")
        print(f"📂 Model saved to: {OUTPUT_DIR}")
        print(f"🔗 Ready for ADK integration!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Check prerequisites
    if not torch.backends.mps.is_available():
        print("⚠️ MPS not available. Training will use CPU (slower).")
        print("💡 Make sure you're using PyTorch with MPS support.")
    
    main()
