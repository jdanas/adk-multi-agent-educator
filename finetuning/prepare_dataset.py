"""
Dataset Preparation for English Agent Fine-tuning

This script helps you prepare and format datasets for fine-tuning
the English language education agent.
"""

import json
import csv
import pandas as pd
from typing import List, Dict, Any

def create_grammar_correction_dataset():
    """Create a dataset focused on grammar correction tasks"""
    
    grammar_examples = [
        {
            "category": "Subject-Verb Agreement",
            "incorrect": "The group of students are studying.",
            "correct": "The group of students is studying.",
            "explanation": "When the subject is a collective noun like 'group', use singular verb form 'is' not 'are'."
        },
        {
            "category": "Pronoun Case",
            "incorrect": "Between you and I, this is difficult.",
            "correct": "Between you and me, this is difficult.", 
            "explanation": "After prepositions like 'between', use object pronouns (me, him, her, us, them), not subject pronouns (I, he, she, we, they)."
        },
        {
            "category": "Apostrophe Usage",
            "incorrect": "The dog wagged it's tail happily.",
            "correct": "The dog wagged its tail happily.",
            "explanation": "'Its' is possessive (no apostrophe). 'It's' is a contraction meaning 'it is'."
        },
        {
            "category": "Dangling Modifiers",
            "incorrect": "Walking to school, the backpack felt heavy.",
            "correct": "Walking to school, Sarah felt her backpack was heavy.",
            "explanation": "The modifier 'walking to school' should clearly refer to who is doing the walking."
        },
        {
            "category": "Run-on Sentences",
            "incorrect": "I love reading books they help me learn new things I read every day.",
            "correct": "I love reading books because they help me learn new things. I read every day.",
            "explanation": "Break run-on sentences into separate sentences or use proper conjunctions."
        }
    ]
    
    return grammar_examples

def create_writing_help_dataset():
    """Create a dataset for writing assistance and improvement"""
    
    writing_examples = [
        {
            "task": "Thesis Statement",
            "weak": "Social media is bad for teenagers.",
            "strong": "While social media provides teenagers with opportunities for connection and self-expression, excessive use significantly impacts their mental health, academic performance, and real-world social development.",
            "tips": [
                "Be specific rather than general",
                "Acknowledge complexity with 'while' or 'although'",
                "Provide a roadmap of main points",
                "Make it arguable - not just a fact"
            ]
        },
        {
            "task": "Paragraph Structure",
            "weak": "Exercise is good. It helps your body. You should do it more. It makes you feel better too.",
            "strong": "Regular exercise provides numerous benefits for both physical and mental health. Physically, consistent exercise strengthens cardiovascular systems, builds muscle mass, and improves flexibility. Moreover, exercise releases endorphins that enhance mood and reduce stress levels. Therefore, incorporating even 30 minutes of daily physical activity can significantly improve overall well-being.",
            "tips": [
                "Start with a clear topic sentence",
                "Use transitions between ideas",
                "Provide specific details and examples",
                "End with a concluding thought"
            ]
        }
    ]
    
    return writing_examples

def create_vocabulary_dataset():
    """Create vocabulary building and word choice dataset"""
    
    vocabulary_examples = [
        {
            "word_pair": "Affect vs Effect",
            "explanation": "AFFECT is a verb meaning to influence. EFFECT is a noun meaning result.",
            "examples": [
                "The rain will affect our picnic plans. (verb)",
                "The effect of rain was a cancelled picnic. (noun)"
            ],
            "memory_trick": "A for Action (affect = verb), E for End result (effect = noun)"
        },
        {
            "word_pair": "Who vs Whom", 
            "explanation": "WHO is subject (does action). WHOM is object (receives action).",
            "examples": [
                "Who is calling? (subject - he is calling)",
                "To whom are you speaking? (object - speaking to him)"
            ],
            "memory_trick": "If you can substitute 'he', use WHO. If you can substitute 'him', use WHOM."
        },
        {
            "word_pair": "Less vs Fewer",
            "explanation": "LESS for uncountable things. FEWER for countable things.",
            "examples": [
                "Less water (uncountable)",
                "Fewer books (countable)"
            ],
            "memory_trick": "If you can count it individually, use FEWER."
        }
    ]
    
    return vocabulary_examples

def format_for_training(data_type: str, examples: List[Dict]) -> List[Dict]:
    """Format examples into training conversation format"""
    
    formatted_examples = []
    
    if data_type == "grammar":
        for example in examples:
            conversation = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a specialized English Language Education Agent focused on grammar instruction. Provide clear explanations with examples."
                    },
                    {
                        "role": "user", 
                        "content": f"Please fix the grammar in this sentence and explain what was wrong: '{example['incorrect']}'"
                    },
                    {
                        "role": "assistant",
                        "content": f"**Corrected sentence:** {example['correct']}\n\n**Grammar rule ({example['category']}):** {example['explanation']}\n\n**Remember:** This is a common mistake, so always double-check for this pattern in your writing!"
                    }
                ]
            }
            formatted_examples.append(conversation)
    
    elif data_type == "writing":
        for example in examples:
            tips_text = "\n".join([f"• {tip}" for tip in example['tips']])
            conversation = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a specialized English Language Education Agent focused on writing improvement. Help students write more effectively."
                    },
                    {
                        "role": "user",
                        "content": f"How can I improve this {example['task'].lower()}? Here's what I have: '{example['weak']}'"
                    },
                    {
                        "role": "assistant", 
                        "content": f"**Improved {example['task']}:** {example['strong']}\n\n**Writing tips for {example['task'].lower()}s:**\n{tips_text}\n\n**Practice:** Try rewriting your original using these principles!"
                    }
                ]
            }
            formatted_examples.append(conversation)
    
    elif data_type == "vocabulary":
        for example in examples:
            examples_text = "\n".join([f"• {ex}" for ex in example['examples']])
            conversation = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a specialized English Language Education Agent focused on vocabulary and word usage. Make confusing word pairs clear."
                    },
                    {
                        "role": "user",
                        "content": f"I'm confused about when to use {example['word_pair']}. Can you help?"
                    },
                    {
                        "role": "assistant",
                        "content": f"**{example['word_pair']}:**\n\n{example['explanation']}\n\n**Examples:**\n{examples_text}\n\n**Memory trick:** {example['memory_trick']}\n\n**Quick test:** Try substituting the words in your own sentences!"
                    }
                ]
            }
            formatted_examples.append(conversation)
    
    return formatted_examples

def save_dataset(formatted_examples: List[Dict], filename: str):
    """Save formatted dataset to JSON file"""
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(formatted_examples, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(formatted_examples)} examples to {filename}")

def create_combined_dataset():
    """Create a comprehensive dataset combining all types"""
    
    print("📚 Creating English Education Dataset...")
    
    # Create individual datasets
    grammar_data = create_grammar_correction_dataset()
    writing_data = create_writing_help_dataset()
    vocabulary_data = create_vocabulary_dataset()
    
    # Format for training
    grammar_formatted = format_for_training("grammar", grammar_data)
    writing_formatted = format_for_training("writing", writing_data)
    vocabulary_formatted = format_for_training("vocabulary", vocabulary_data)
    
    # Combine all examples
    all_examples = grammar_formatted + writing_formatted + vocabulary_formatted
    
    # Save individual datasets
    save_dataset(grammar_formatted, "grammar_correction_dataset.json")
    save_dataset(writing_formatted, "writing_improvement_dataset.json")
    save_dataset(vocabulary_formatted, "vocabulary_dataset.json")
    
    # Save combined dataset
    save_dataset(all_examples, "english_education_combined_dataset.json")
    
    return all_examples

def load_external_datasets():
    """Guide for loading external educational datasets"""
    
    external_sources = {
        "Grammar Correction": [
            "jfleg - Fluency correction dataset",
            "lang8 - Learner error correction",
            "wi-locness - Write & Improve dataset"
        ],
        "Writing Improvement": [
            "essay_scoring datasets",
            "persuasive_essays datasets",
            "writing_prompts datasets"
        ],
        "Vocabulary": [
            "wordnet - Word definitions and relations",
            "oxford_dictionaries - Usage examples",
            "vocabulary_exercises datasets"
        ]
    }
    
    print("🔗 External Dataset Recommendations:")
    for category, sources in external_sources.items():
        print(f"\n📖 {category}:")
        for source in sources:
            print(f"   • {source}")
    
    print(f"\n💡 To use external datasets:")
    print(f"   1. Load with: dataset = load_dataset('dataset_name')")
    print(f"   2. Format using the format_for_training() function")
    print(f"   3. Combine with your custom examples")

if __name__ == "__main__":
    print("🎓 English Agent Dataset Preparation")
    print("=" * 50)
    
    # Create datasets
    dataset = create_combined_dataset()
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Total examples: {len(dataset)}")
    print(f"   Grammar examples: {len([ex for ex in dataset if 'grammar' in str(ex).lower()])}")
    print(f"   Writing examples: {len([ex for ex in dataset if 'writing' in str(ex).lower() or 'thesis' in str(ex).lower()])}")
    print(f"   Vocabulary examples: {len([ex for ex in dataset if 'vs' in str(ex).lower() or 'affect' in str(ex).lower()])}")
    
    # Show external dataset info
    print("\n" + "=" * 50)
    load_external_datasets()
    
    print(f"\n✅ Dataset preparation complete!")
    print(f"📁 Files created:")
    print(f"   • english_education_combined_dataset.json")
    print(f"   • grammar_correction_dataset.json") 
    print(f"   • writing_improvement_dataset.json")
    print(f"   • vocabulary_dataset.json")
    print(f"\n🚀 Ready for fine-tuning with Unsloth!")
