"""
Fine-tuning English Agent with Unsloth for Educational Grammar Support

This script demonstrates how to fine-tune a language model using Unsloth
specifically for educational English language and grammar tasks.
"""

import torch
from unsloth import FastLanguageModel, FastModel
from datasets import Dataset, load_dataset
from trl import SFTTrainer, SFTConfig
import json
import os

# Configuration
MAX_SEQ_LENGTH = 2048
MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"  # Fast for educational use
OUTPUT_DIR = "./english_agent_finetuned"

def setup_model():
    """Setup and configure the model for fine-tuning"""
    print("🔄 Loading model and tokenizer...")
    
    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,  # 4-bit quantization for memory efficiency
        dtype=None,  # Auto-detect
    )
    
    # Add LoRA adapters for efficient fine-tuning
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Rank for LoRA
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha=16,
        lora_dropout=0,  # Optimized for Unsloth
        bias="none",     # Optimized for Unsloth
        use_gradient_checkpointing="unsloth",  # Memory efficient
        random_state=3407,
        max_seq_length=MAX_SEQ_LENGTH,
        use_rslora=False,
    )
    
    return model, tokenizer

def create_english_education_dataset():
    """Create a specialized dataset for English education"""
    
    # Educational English conversation examples
    educational_examples = [
        {
            "instruction": "Help me fix the grammar in this sentence and explain what was wrong.",
            "input": "Me and my friend went to store yesterday.",
            "output": "**Corrected sentence:** My friend and I went to the store yesterday.\n\n**Grammar explanation:**\n1. Use 'My friend and I' instead of 'Me and my friend' when it's the subject\n2. Add the article 'the' before 'store'\n3. Rule: When referring to yourself and others as the subject, put yourself last and use 'I' not 'me'"
        },
        {
            "instruction": "Explain the difference between affect and effect with examples.",
            "input": "I'm confused about when to use affect vs effect.",
            "output": "**AFFECT vs EFFECT:**\n\n**Affect** (verb) = to influence or change something\n- Example: 'The rain will affect our picnic plans.'\n- Memory trick: **A**ffect = **A**ction (verb)\n\n**Effect** (noun) = the result or consequence\n- Example: 'The effect of rain was a cancelled picnic.'\n- Memory trick: **E**ffect = **E**nd result (noun)\n\n**Quick test:** Try replacing with 'influence' (affect) or 'result' (effect)"
        },
        {
            "instruction": "Help me understand when to use who vs whom.",
            "input": "Who/whom should I give this to?",
            "output": "**Correct:** Whom should I give this to?\n\n**WHO vs WHOM rule:**\n- **Who** = subject (does the action)\n- **Whom** = object (receives the action)\n\n**Test:** Replace with he/him\n- If 'he' fits → use WHO\n- If 'him' fits → use WHOM\n\n**Your sentence:** 'I should give this to him' → use WHOM\n\n**Examples:**\n- Who is calling? (He is calling)\n- To whom are you speaking? (You are speaking to him)"
        },
        {
            "instruction": "Explain what a dangling modifier is and fix this sentence.",
            "input": "Running quickly, the bus was missed.",
            "output": "**Problem:** This is a dangling modifier! The sentence suggests the bus was running.\n\n**Fixed sentence:** Running quickly, I missed the bus.\n\n**What's a dangling modifier?**\nA modifier that doesn't clearly connect to what it's describing.\n\n**How to fix:**\n1. Make sure the modifier describes the subject immediately after the comma\n2. Rewrite to clarify who/what is doing the action\n\n**More examples:**\n- Wrong: 'After studying all night, the test was easy.'\n- Right: 'After studying all night, I found the test easy.'"
        },
        {
            "instruction": "Help me write a strong thesis statement for my essay about social media.",
            "input": "I want to write about how social media affects teenagers.",
            "output": "**Strong thesis statement:**\n'While social media provides teenagers with valuable opportunities for connection and self-expression, its excessive use significantly impacts their mental health, academic performance, and real-world social skills.'\n\n**Why this works:**\n1. **Specific topic:** Social media and teenagers\n2. **Clear position:** Acknowledges benefits but focuses on negative impacts\n3. **Roadmap:** Lists 3 main effects to discuss\n4. **Arguable:** Someone could disagree\n\n**Thesis formula:**\n[Topic] + [Your position] + [Main supporting points] = Strong thesis\n\n**Make it even stronger by adding:**\n- Specific statistics or timeframes\n- More precise language"
        },
        {
            "instruction": "Explain the difference between active and passive voice.",
            "input": "When should I use active vs passive voice in writing?",
            "output": "**ACTIVE vs PASSIVE VOICE:**\n\n**Active voice:** Subject does the action\n- Formula: Subject + Verb + Object\n- Example: 'The teacher graded the tests.'\n- **Use when:** You want clear, direct, engaging writing\n\n**Passive voice:** Action is done to the subject\n- Formula: Object + 'to be' + Past participle + (by Subject)\n- Example: 'The tests were graded by the teacher.'\n- **Use when:** The doer is unknown, unimportant, or you want to emphasize the action\n\n**Generally prefer active because it's:**\n- Clearer and more direct\n- Uses fewer words\n- More engaging\n\n**Passive is okay for:**\n- Scientific writing\n- When the doer is obvious or unknown"
        }
    ]
    
    # Convert to Hugging Face dataset format
    formatted_examples = []
    for example in educational_examples:
        # Format as conversation with system prompt
        conversation = f"""<|im_start|>system
You are a specialized English Language Education Agent. Your role is to help students with grammar, writing, vocabulary, and language concepts. Always provide clear explanations with examples and teaching principles.<|im_end|>
<|im_start|>user
{example['instruction']}
{example['input']}<|im_end|>
<|im_start|>assistant
{example['output']}<|im_end|>"""
        
        formatted_examples.append({"text": conversation})
    
    # You can also load external datasets for more variety
    try:
        # Example: Load grammar correction dataset
        external_dataset = load_dataset("jfleg", split="train[:100]")  # Small sample
        
        for item in external_dataset:
            if 'sentence' in item and 'corrections' in item:
                correction = item['corrections'][0] if item['corrections'] else item['sentence']
                conversation = f"""<|im_start|>system
You are a specialized English Language Education Agent. Your role is to help students with grammar, writing, vocabulary, and language concepts.<|im_end|>
<|im_start|>user
Please fix the grammar in this sentence and explain what was wrong: {item['sentence']}<|im_end|>
<|im_start|>assistant
**Corrected sentence:** {correction}

**Grammar explanation:** This sentence has been corrected for proper grammar, punctuation, and clarity. The main issues were likely related to sentence structure, word choice, or punctuation usage.<|im_end|>"""
                
                formatted_examples.append({"text": conversation})
                
    except Exception as e:
        print(f"⚠️ Could not load external dataset: {e}")
        print("📝 Using only manual examples")
    
    return Dataset.from_list(formatted_examples)

def fine_tune_english_agent():
    """Main fine-tuning function"""
    print("🎓 Starting English Agent Fine-tuning with Unsloth")
    
    # Setup model
    model, tokenizer = setup_model()
    
    # Create dataset
    print("📚 Creating educational dataset...")
    dataset = create_english_education_dataset()
    print(f"📊 Dataset size: {len(dataset)} examples")
    
    # Configure training
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=SFTConfig(
            per_device_train_batch_size=1,  # Adjust based on GPU memory
            gradient_accumulation_steps=4,   # Effective batch size = 4
            warmup_steps=5,
            max_steps=50,  # Increase for more training
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=OUTPUT_DIR,
            save_steps=25,
            save_total_limit=2,
        ),
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=1,  # Important for Windows compatibility
    )
    
    # Start training
    print("🚀 Starting training...")
    trainer.train()
    
    # Save the model
    print("💾 Saving fine-tuned model...")
    model.save_pretrained(f"{OUTPUT_DIR}/final_model")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_model")
    
    print("✅ Fine-tuning complete!")
    return model, tokenizer

def test_finetuned_model(model, tokenizer):
    """Test the fine-tuned model with sample questions"""
    print("\n🧪 Testing fine-tuned English Agent...")
    
    # Enable inference mode
    FastLanguageModel.for_inference(model)
    
    test_questions = [
        "Help me fix this sentence: 'The book who I read was interesting.'",
        "Explain the difference between 'your' and 'you're'",
        "What's wrong with this sentence: 'Between you and I, this is difficult.'"
    ]
    
    for question in test_questions:
        print(f"\n❓ Question: {question}")
        
        # Format input
        messages = [
            {"role": "system", "content": "You are a specialized English Language Education Agent."},
            {"role": "user", "content": question}
        ]
        
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        print(f"🤖 Response: {response}")

def save_to_gguf():
    """Save the model in GGUF format for efficient deployment"""
    print("📦 Saving to GGUF format...")
    
    # This requires additional setup - see Unsloth docs
    # model.save_pretrained_gguf("english_agent_model", tokenizer, quantization_method="q4_k_m")
    
    print("💡 To save as GGUF, follow the Unsloth GGUF guide:")
    print("   https://github.com/unslothai/unsloth/wiki")

if __name__ == "__main__":
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available. This script requires a GPU.")
        exit(1)
    
    print(f"🔥 Using GPU: {torch.cuda.get_device_name()}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Fine-tune the model
    model, tokenizer = fine_tune_english_agent()
    
    # Test the model
    test_finetuned_model(model, tokenizer)
    
    # Optional: Save to GGUF
    # save_to_gguf()
    
    print("\n🎉 English Agent fine-tuning complete!")
    print(f"📁 Model saved to: {OUTPUT_DIR}")
    print("🚀 You can now integrate this with your ADK system!")
