import re
from openai import OpenAI
import base64
from io import BytesIO
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import argparse
from PIL import Image
import os
import pandas as pd
from vllm.sampling_params import GuidedDecodingParams
from instr_tuning.utils import extract_answer, apply_chat_template, prepare_input_boxed, QWQ_USER_PROMPT_REWARD, QWQ_USER_PROMPT, extract_correctness, GRAMMAR, USER_PROMPT_DETAILED_AND_LOCALIZATION, PATTERN_DETAILED_AND_LOCALIZATION

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
MODEL_NAME = "Qwen/QwQ-32B-AWQ"

from collections import Counter
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

import torch

def extract_answer(solution_text: str):
    boxed_pattern = r'\\boxed\{([^}]*)\}'
    matches = re.findall(boxed_pattern, solution_text)
    if matches:
        return matches[-1].strip()
    return None

def apply_chat_template(toker, messages):
    input_prompt = toker.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return toker(input_prompt, add_special_tokens=False).input_ids

def prepare_input_boxed(template, input_d):
    problem = input_d['problem']
    steps = input_d['steps']
    tagged_response = ''
    for sdx, step in enumerate(steps):
        tagged_response += f'<paragraph_{sdx}>\n{step}\n</paragraph_{sdx}>\n\n'
    tagged_response = tagged_response.strip()
    prompt = template.format(problem=problem, tagged_response=tagged_response)
    messages = [{'role': 'user', 'content': prompt}]
    return messages

if __name__ == "__main__":
    llm = LLM(
        model=MODEL_NAME, tokenizer=MODEL_NAME,
        gpu_memory_utilization=0.8,
        tensor_parallel_size=torch.cuda.device_count(),
        enable_prefix_caching=True,
    )
    tokenizer = llm.get_tokenizer()

    translated_train_answer = pd.read_csv("./instr_tuning/data/serve_question/question_data/train_data_QA_trans.csv", encoding="utf-8-sig")

    with open("./instr_tuning/data/processed_data_wo_korean.json", "r", encoding="utf-8-sig") as f:
        processed_train_data = json.load(f)

    bs = 64
    max_token = 512
    end_range = 64 * 160 // bs

    qwq_responses = []
    current_data = range(end_range)
    for i in tqdm(current_data):
        all_good = False
        run_cnt = 0
        batch = processed_train_data[i * bs:(i + 1) * bs]
        defective = [x for x in range(bs)]
        current_responses = [None for _ in range(bs)]
        while not all_good and run_cnt < 5:
            all_good = True

            b = [content for idx, content in enumerate(batch) if idx in defective]
            idxs = [content["question_filename"][:-4] for content in b]
            qa = [translated_train_answer[translated_train_answer["id"] == idx] for idx in idxs]
            qa = pd.concat(qa, ignore_index=True)
            questions = qa["question_en"].values
            references = qa["answer_en"].values
            students = [content["explanation_info"][0]["explanation_text"] for content in b]

            messages = [[
                {'role': 'user', 'content': QWQ_USER_PROMPT.format(
                    problem=question,
                    reference=reference,
                    student=student,
                    # error_location=error_location,
                )},
            ] for question, reference, student in zip(questions, references, students)]

            sampling_params = SamplingParams(
                temperature=0.8,
                top_p=0.9,
                top_k=40,
                max_tokens=max_token,
                stop=["</localization>"]
            )
            completions = llm.generate(
                prompts=tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False),
                sampling_params=sampling_params,
                use_tqdm=False,
            )

            for c_id, completion in enumerate(completions):
                output = completion.outputs[0]
                msg = messages[c_id]
                if len(output.token_ids) < max_token:
                    current_responses[c_id] = "<think>\n" + output.text + "</localization>"
                else:
                    # If the output is at max_token, we forcefully add </think> and let the model finish the response
                    prompt: str = tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
                    end_think: str = "\n\n".join(output.text.split("\n\n")[:-1]) + "\n</think>\n\n<correctness>"
                    sampling_params.guided_decoding = GuidedDecodingParams(
                        grammar=GRAMMAR,
                    )
                    sampling_params.max_tokens = max_token // 2
                    force_completions = llm.generate(
                        prompts=prompt + end_think,
                        sampling_params=sampling_params,
                        use_tqdm=False,
                    )
                    final_response = force_completions[0].outputs[0].text
                    current_responses[c_id] = "<think>\n" + end_think + final_response + "</localization>"
                current_response = current_responses[c_id]
                correctness = extract_correctness(current_response)
                if correctness not in ["correct", "incorrect"]:
                    all_good = False
                    current_responses[c_id] = None
                    
            defective = [idx for idx, response in enumerate(current_responses) if response is None]
            if len(defective) == 0:
                break

            run_cnt += 1

        qwq_responses.extend(current_responses)

    filtered_responses = [response for response in qwq_responses if response is not None and response.endswith("</localization>")]
    len(filtered_responses)
        
    # Save the responses to a file
    with open("qwq_generated_responses.json", "w", encoding="utf-8") as f:
        json.dump(filtered_responses, f, ensure_ascii=False, indent=4)