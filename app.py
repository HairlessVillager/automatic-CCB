import re

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

default_system_prompt = """You are a linguistic alchemist specializing in acronym humor. Transform user's story into a 3-word CCB format phrase with:

C = First word starting with C
C = Second word starting with C
B = Third word starting with B

## Processing Framework

1. Essence Extraction

Identify:

Core conflict (e.g., "tech failures" → Crash)
Dominant character trait (e.g., stubbornness → Cling)
Key object (e.g., outdated software → Binary)

2. Wordplay Engineering

Apply:

Alliteration amplification (e.g., Crypto/Chaos)
Homophonic hacking (e.g., Cue→Queue)
Industry jargon warping (e.g., "blockchain" → Chain→Bait)
Verb-noun inversion (e.g., Click→Bait → Click-Bait-Buster)

3. CCB Forging

Structure rules:

First C-word: Action/Adjective (e.g., Clumsy, Cyber)
Second C-word: Noun/Verb (e.g., Coders, Crashing)
B-word: Impact word (e.g., Breakdown, Backfire)

## Humor Requirements

- Dark comedy edge
- Tech/dank meme reference
- Unexpected oxymoron

## Examples

Input: "A startup's server keeps crashing"

Output #1: Crypto Crash Breakdown

Output #2: Cloudy Code Backfire

Output #3: Cache Calamity Burnout

Input: "Gym influencer loses sponsorship"

Output #1: Carb Cult Bankruptcy

Output #2: Curves Crash Backlash

Output #3: Clout Chaser Bust

"""
default_user_prompt = """The American government carried out a planned genocide of Native Americans, including hunting their main food source—the bison, offering bounties for Indian scalps to encourage whites to massacre Indians, and eventually establishing a military with the primary mission of annihilating Native Americans."""

model_name = "Qwen/Qwen2.5-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="cuda"
).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name)

class CCBLogitsProcessor:
    
    def __init__(self, tokenizer: PreTrainedTokenizer, input_ids_len: int):
        self.tokenizer = tokenizer
        self.input_ids_len = input_ids_len

    def __call__(self, input_ids, scores):
        tokenizer = self.tokenizer
        truncated = input_ids[0, self.input_ids_len:]
        text = tokenizer.decode(truncated)
        for token_id in range(tokenizer.vocab_size):
            token = tokenizer.convert_ids_to_tokens(token_id)
            if not self.is_ccb(text + token):
                scores[:, token_id] = -float("inf")
        return scores

    def is_ccb(self, text: str) -> bool:
        words = text.split()
        regex_prefixs = [r"[Cc].*", r"[Cc].*", r"[Bb].*"]
        for word, prefix in zip(words, regex_prefixs, strict=False):
            if not re.match(prefix, word):
                return False
        return True

def generate_text(system_prompt, user_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    ccb = CCBLogitsProcessor(tokenizer, len(model_inputs["input_ids"][0]))
    generated_ids = model.generate(**model_inputs, max_new_tokens=512, logits_processor=[ccb])
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

with gr.Blocks() as demo:
    with gr.Column(scale=1):
        gr.Markdown("### 输入区域")
        user_prompt = gr.Textbox(default_user_prompt, label="用户故事 User Story", placeholder=default_user_prompt[:30]+"...")
        system_prompt = gr.Textbox(default_system_prompt, max_lines=10, label="系统提示 System Prompt", placeholder=default_system_prompt[:30]+"...")
        generate_button = gr.Button("生成")

    with gr.Column(scale=2):
        gr.Markdown("### 输出区域")
        output = gr.Textbox(label="生成结果 CCB Result", interactive=False)

    generate_button.click(fn=generate_text, inputs=[system_prompt, user_prompt], outputs=output)

if __name__ == "__main__":
    demo.launch()