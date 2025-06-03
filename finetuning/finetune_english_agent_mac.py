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

def setup_lora_config():
    """Configure LoRA for efficient fine-tuning"""
    print("⚙️ Setting up LoRA configuration...")
    
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,  # Rank - balance between efficiency and quality
        lora_alpha=32,  # Scaling factor
        lora_dropout=0.1,  # Regularization
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Attention layers
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
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)
    
    print(f"📈 Trainable parameters: {model.print_trainable_parameters()}")
    
    # Training arguments optimized for Mac M4
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
        report_to=None,  # Disable wandb/tensorboard
        load_best_model_at_end=False,
        evaluation_strategy="no",
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
