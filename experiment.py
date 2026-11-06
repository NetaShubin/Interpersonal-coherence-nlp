import os
import gc
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# --- ROLE-ANCHORED UNBIASED PROMPT ---
# Directs the model to anchor to the established professor identity from context
# without prescribing explicit stylistic traits (e.g., forcing analogies or Socratic questions).
MINIMAL_PROFESSOR_PROMPT = (
    "You are participating in an academic dialogue. Continue the conversation "
    "by responding to the student's latest question as the philosophy professor "
    "from earlier in the conversation."
)

def initialize_embedder():
    print("Loading local semantic embedding model (all-MiniLM-L6-v2)...")
    return SentenceTransformer('all-MiniLM-L6-v2')

def evaluate_dynamic_turn(embedder, dynamic_history, gold_response, generated_response):
    """
    Evaluates dynamic rollout across 5 core dimensions:
    1. Contextual Semantic Alignment (cosine similarity against dynamic history)
    2. Rhetorical Marker Density (Concession & Justification counts)
    3. Lexical Continuity (vocabulary overlap with dynamic history)
    4. Verbosity Ratio (length relative to ground-truth professor response)
    5. Interrogative / Socratic Density
    """
    CONCESSION_MARKERS = [
        "however", "nevertheless", "granted", "admittedly", "conversely",
        "although", "even though", "notwithstanding", "albeit", "despite",
        "on the other hand", "be that as it may", "while it is true",
        "having said that", "at the same time", "in spite of", "still",
        "alternatively", "that being said", "even so", "regardless"
    ]
    JUSTIFICATION_MARKERS = [
        "because", "therefore", "consequently", "thus", "since", "hence",
        "it follows that", "as a result", "accordingly", "ergo", "for this reason",
        "by virtue of", "this entails", "deductively", "on the grounds that", "for example",
        "specifically", "in virtue of", "given that", "seeing that", "owing to",
        "this implies", "inasmuch as", "subsequently", "thereby", "to this end"
    ]

    # 1. Contextual Semantic Alignment against cumulative dynamic history
    if not generated_response.strip():
        similarity = 0.0
    else:
        history_text = " ".join([m["content"] for m in dynamic_history if m["role"] != "system"])
        embeddings = embedder.encode([history_text, generated_response], convert_to_tensor=True)
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()

    # 2. Rhetorical Markers
    gen_lower = generated_response.lower()
    concession_count = sum(1 for word in CONCESSION_MARKERS if word in gen_lower)
    justification_count = sum(1 for word in JUSTIFICATION_MARKERS if word in gen_lower)

    # 3. Lexical Continuity (overlap with running dynamic history)
    context_vocab = set(history_text.lower().split())
    gen_vocab = set(gen_lower.split())
    lexical_overlap = len(context_vocab.intersection(gen_vocab)) / len(gen_vocab) if gen_vocab else 0.0

    # 4. Verbosity Ratio (relative to gold professor turn length)
    gold_words = len(gold_response.split())
    gen_words = len(generated_response.split())
    verbosity_ratio = (gen_words / gold_words) if gold_words > 0 else 0.0

    # 5. Socratic / Interrogative Density
    question_count = generated_response.count("?")
    socratic_phrases = [
        "suppose", "consider", "what if", "does", "can we", "how do",
        "do you think", "if", "can", "could", "how", "why", "what",
        "is it", "would", "shall", "might", "to what extent", "in what way"
    ]
    socratic_phrase_count = sum(1 for phrase in socratic_phrases if phrase in gen_lower)
    interrogative_score = question_count + socratic_phrase_count

    return {
        "semantic_similarity": round(similarity, 4),
        "concession_markers": concession_count,
        "justification_markers": justification_count,
        "lexical_overlap": round(lexical_overlap, 4),
        "verbosity_ratio": round(verbosity_ratio, 4),
        "interrogative_density": interrogative_score
    }

def run_large_scale_rollout_experiment():
    output_dir = "./carc_outputs_revamped"
    os.makedirs(output_dir, exist_ok=True)

    summary_csv = os.path.join(output_dir, "rollout_100conv_summary_results.csv")
    individual_csv = os.path.join(output_dir, "rollout_100conv_detailed_responses.csv")

    embedder = initialize_embedder()

    print("Loading dataset...")
    dataset = load_dataset("ruggsea/stanford-encyclopedia-of-philosophy_chat_multi_turn", split="train")

    # Filter for conversations with at least 16 message turns
    eligible_data = [
        item for item in dataset
        if isinstance(item.get("conversation"), list) and len(item.get("conversation")) >= 16
    ]

    print(f"Total eligible conversations found (>= 16 turns): {len(eligible_data)}")

    # Sort in descending order of length to prioritize the longest available dialogues
    eligible_data.sort(key=lambda x: len(x["conversation"]), reverse=True)

    # Select top 100 longest conversations
    NUM_TARGET_SAMPLES = 100
    target_samples = eligible_data[:NUM_TARGET_SAMPLES]

    print(f"Selected {len(target_samples)} longest conversations for rollout evaluation.")
    print(f" - Longest conversation in sample: {len(target_samples[0]['conversation'])} turns")
    print(f" - Shortest conversation in sample: {len(target_samples[-1]['conversation'])} turns")

    max_turn_cap = 25  # Rollout cap

    MODELS_TO_TEST = [
        "Qwen/Qwen2.5-1.5B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct"
    ]

    for model_name in MODELS_TO_TEST:
        print(f"\n==========================================")
        print(f"Starting Rollout Experiment for: {model_name}")
        print(f"==========================================")
        
        try:
            gen_pipeline = pipeline(
                "text-generation", 
                model=model_name, 
                device_map="auto", 
                torch_dtype=torch.float16
            )
        except Exception as e:
            print(f"Failed to load model {model_name}: {e}. Skipping...")
            continue

        for idx, item in enumerate(tqdm(target_samples, desc=f"Evaluating {model_name}")):
            gold_conv = item["conversation"]

            # Warm-up Phase: Initialize dynamic history with role prompt + real turns 0 to 4
            dynamic_history = [{"role": "system", "content": MINIMAL_PROFESSOR_PROMPT}]
            for turn in gold_conv[:5]:
                dynamic_history.append({"role": turn.get("role", "user"), "content": turn.get("content", "")})

            conversation_results = []
            detailed_response_results = []
            
            max_turn = min(len(gold_conv) - 1, max_turn_cap)

            # Dynamic Rollout Loop (Turn 5 up to max_turn)
            for t in range(5, max_turn + 1):
                # Evaluate only when target index corresponds to an assistant (professor) turn
                if gold_conv[t].get("role") != "assistant":
                    continue

                # Inject the student's question for turn t (index t-1 in dataset)
                if t > 5:
                    student_text = gold_conv[t - 1]["content"]
                    dynamic_history.append({"role": "user", "content": student_text})

                gold_response = gold_conv[t]["content"]

                try:
                    prompt_text = gen_pipeline.tokenizer.apply_chat_template(
                        dynamic_history, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                    outputs = gen_pipeline(
                        prompt_text, 
                        max_new_tokens=150, 
                        temperature=0.2, 
                        do_sample=True, 
                        pad_token_id=gen_pipeline.tokenizer.eos_token_id
                    )
                    gen_response = outputs[0]["generated_text"][len(prompt_text):].strip()
                except Exception as e:
                    print(f"Error during generation at conv {idx}, turn {t}: {e}")
                    gen_response = ""

                # Evaluate performance metrics against dynamic history
                metrics = evaluate_dynamic_turn(embedder, dynamic_history, gold_response, gen_response)

                # DYNAMIC ROLLOUT STEP: Append model's output back to history for turn t+1
                dynamic_history.append({"role": "assistant", "content": gen_response})

                conversation_results.append({
                    "model_name": model_name,
                    "conversation_id": idx,
                    "turn_depth": t,
                    **metrics,
                    "generated_snippet": gen_response[:100]
                })

                detailed_response_results.append({
                    "model_name": model_name,
                    "conversation_id": idx,
                    "turn_depth": t,
                    "gold_response": gold_response,
                    "generated_response": gen_response,
                    **metrics
                })

            # Save summary metrics incrementally after each conversation
            if conversation_results:
                df_conv = pd.DataFrame(conversation_results)
                if not os.path.exists(summary_csv):
                    df_conv.to_csv(summary_csv, index=False)
                else:
                    df_conv.to_csv(summary_csv, mode='a', header=False, index=False)

            # Save detailed response text incrementally after each conversation
            if detailed_response_results:
                df_detailed = pd.DataFrame(detailed_response_results)
                if not os.path.exists(individual_csv):
                    df_detailed.to_csv(individual_csv, index=False)
                else:
                    df_detailed.to_csv(individual_csv, mode='a', header=False, index=False)

        # Clear VRAM before loading the next model
        del gen_pipeline
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\n==========================================")
    print(f"Large-Scale Dynamic Rollout Experiment Complete!")
    print(f" - Metric Summary CSV saved to: {summary_csv}")
    print(f" - Individual Responses CSV saved to: {individual_csv}")
    print(f"==========================================")

if __name__ == "__main__":
    run_large_scale_rollout_experiment()
