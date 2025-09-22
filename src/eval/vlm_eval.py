import math
import re
from src.prompts.localization import SYSTEM_PROMPT, PATTERN_DETAILED_AND_LOCALIZATION, FERMAT_USER_PROMPT_DETAILED_AND_LOCALIZATION_FOR_EVAL, USER_PROMPT_DETAILED_AND_LOCALIZATION_FOR_EVAL
from openai import OpenAI
import base64
from io import BytesIO
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import argparse
from PIL import Image
import os
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, average_precision_score

client = OpenAI(
    base_url="http://__YOUR_IP_ADDRESS__:__YOUR_PORT__/v1",
    api_key="__YOUR_API_KEY__",
)

MAX_RATIO = 200

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 4*28*28, max_pixels: int = 64*28*28
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, floor_by_factor(height / beta, factor))
        w_bar = max(factor, floor_by_factor(width / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

def encode_image_to_base64(image):
    """Encodes a PIL image to a Base64 string."""
    if isinstance(image, dict):
        image = image["path"]
    with open(image, "rb") as img_file:
        image = Image.open(img_file)
        image = image.convert("RGB")
    
    # # Resize the image to 224x224
    # image = image.resize((224, 224))
    new_h, new_w = smart_resize(image.size[1], image.size[0])
    image = image.resize((new_w, new_h))
    # image = image.resize((448, 448))
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def resize_from_base64(image_base64):
    """Resizes a Base64 image to 224x224."""
    begin_text = "data:image/jpeg;base64,"
    if image_base64.startswith(begin_text):
        image_base64 = image_base64[len(begin_text):]
    image_data = base64.b64decode(image_base64)
    image = Image.open(BytesIO(image_data))
    image = image.convert("RGB")
    
    # # Resize the image to 224x224
    # new_h, new_w = smart_resize(image.size[1], image.size[0])
    # image = image.resize((new_w, new_h))
    # image = image.resize((448, 448))
    
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return begin_text + base64.b64encode(buffered.getvalue()).decode("utf-8")


def format_data(sample):
    instr = sample["messages"][0]["content"]
    ref_str = "Reference Answer: "
    student_str = "Student Answer: "
    ref_ans_pos = instr.find(ref_str) + len(ref_str)
    reference_answer = instr[ref_ans_pos:].split(student_str)[0]
    conversation = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
    ]
    if "isBase64" in sample and sample["isBase64"]:
        reference_answer = sample["reference_answer"]
        conversation.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": FERMAT_USER_PROMPT_DETAILED_AND_LOCALIZATION_FOR_EVAL,
                },
                {
                    "type": "text",
                    "text": "Question and Student Answer: ",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": resize_from_base64(sample["images"][0]),
                    },
                },
                {
                    "type": "text",
                    "text": f"\nReference Answer: {reference_answer}",
                },
                {
                    "type": "text",
                    "text": "\n",
                },
            ],
        })
    else:
        conversation.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": USER_PROMPT_DETAILED_AND_LOCALIZATION_FOR_EVAL,
                },
                {
                    "type": "text",
                    "text": "Question: ",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image_to_base64(sample['images'][0])}",
                    },
                },
                {
                    "type": "text",
                    "text": f"\nReference Answer: {reference_answer}",
                },
                {
                    "type": "text",
                    "text": "Student Answer: ",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": sample["images"][0] if "isBase64" in sample and sample["isBase64"] else f"data:image/jpeg;base64,{encode_image_to_base64(sample['images'][1])}",
                    },
                },
                {
                    "type": "text",
                    "text": "\n",
                },
            ],
        })
    return conversation


def generate_text_from_sample(model_id, sample):
    samples = format_data(sample)


    completions = client.chat.completions.create(
        model=model_id,
        messages=samples,
        temperature=0.8,
        max_tokens=1024,
        n=1,
    )
    
    # Extract the generated text from the response
    generated_text = completions.choices[0].message.content
    return generated_text

def load_dataset(dataset_path):
    """Load the dataset from the specified path of jsonl file."""
    with open(dataset_path, "r") as f:
        dataset = [json.loads(line) for line in f]
    return dataset

def extract_correctness(response: str):
    """Extract correctness from the response using regex."""
    matches = re.search(PATTERN_DETAILED_AND_LOCALIZATION, response, re.DOTALL | re.MULTILINE)
    correctness = matches.group(1).strip().lower() if matches else ""
    return correctness

def evaluate(response: str, label: str):
    """Evaluate the correctness of the response against the label."""
    correctness = extract_correctness(response)
    return correctness == label.lower()

def calculate_accuracy(results):
    """Calculate the accuracy of the model predictions."""
    correct_predictions = sum(results)
    total_predictions = len(results)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return accuracy

def calculate_precision(results):
    """Calculate the precision of the model predictions."""
    true_positives = sum(1 for result in results if result == "correct")
    false_positives = sum(1 for result in results if result == "incorrect")
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    return precision

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Inference script for multimodal model.")
    parser.add_argument("--base_url", type=str, default="http://localhost", help="base url for inference server")
    parser.add_argument("--port", type=int, default=8000, help="port number for inference server")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Model ID")
    parser.add_argument("--dataset", type=str, default="src/data/test_data_wo_korean_red_bbox.jsonl", help="Dataset path")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for requesting")
    parser.add_argument("--save_path", type=str, default="src/eval/results/aihub", help="Path to save the inference results")

    args = parser.parse_args()

    client = OpenAI(
        base_url=f"{args.base_url}:{args.port}/v1",
        api_key="__YOUR_API_KEY__",
    )

    print("Loading dataset...")
    test_dataset = load_dataset(args.dataset)
    
    def process_data_point(data_point):
        predict = generate_text_from_sample(args.model, data_point)
        return {
            "data": data_point,
            "label": data_point["labels"],
            "predict": predict,
            "result": evaluate(predict, data_point["labels"]),
        }
    
    print("Starting inference...")
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        inference_result = list(tqdm(executor.map(process_data_point, test_dataset), total=len(test_dataset)))

    # Save the inference results to a file
    folder_name = args.model.split("/")[-1]
    save_dir = os.path.join(args.save_path, folder_name)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "inference_result.json"), "w") as f:
        json.dump(inference_result, f, indent=4)

    print("Inference results saved to", args.save_path)

    print("Evaluating results...")
    y_true = []
    y_pred = []
    for item in tqdm(inference_result):
        label = item["label"]
        predict = item["predict"]
        result = item["result"]
        y_true.append(0 if label == "Incorrect" else 1)

        correctness = extract_correctness(predict)
        if len(correctness) == 0:
            y_pred.append(2) 
            continue

        correctness = correctness.lower()
        y_pred.append(0 if correctness == "incorrect" else 1)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    with open(os.path.join(save_dir, "metrics.csv"), "w") as f:
        f.write("Accuracy,Precision,Recall,F1 Score\n")
        f.write(f"{accuracy},{precision},{recall},{f1}\n")
    print(f"Metrics saved to {os.path.join(save_dir, 'metrics.csv')}")
    print("Evaluation completed.")