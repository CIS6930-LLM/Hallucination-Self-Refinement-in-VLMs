#usage python process_haloquest.py --csv-file ... --image-dir ... --output-file ... --ensure-format
import argparse
import csv
import os
import torch
from pathlib import Path

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image
import re


def load_image(image_file):
    """Load image from local file path"""
    if not os.path.exists(image_file):
        raise FileNotFoundError(f"Image not found: {image_file}")
    image = Image.open(image_file).convert("RGB")
    return image


def format_response(response):
    """Ensure response follows Answer: ... Rationale: ... format"""
    response = response.strip()
    
    # Check if already in correct format
    if 'Answer:' in response and 'Rationale:' in response:
        # Try to extract and clean up
        parts = response.split('Rationale:', 1)
        if len(parts) == 2:
            answer_part = parts[0].replace('Answer:', '').strip()
            rationale_part = parts[1].strip()
            return f"Answer: {answer_part}\nRationale: {rationale_part}"
        return response
    
    # If not in format, try to parse and reformat
    # Check if it starts with "Answer:" but missing Rationale
    if response.startswith('Answer:') or 'Answer:' in response:
        answer_part = response.split('Answer:')[-1].strip()
        if 'Rationale:' not in answer_part:
            # Use the answer as rationale if no separate rationale provided
            return f"Answer: {answer_part}\nRationale: Based on the image, {answer_part.lower()}"
    
    # If no format detected, treat entire response as answer
    # and generate a simple rationale
    return f"Answer: {response}\nRationale: Based on visual analysis of the image, {response.lower()}"


def eval_model_single(tokenizer, model, image_processor, args, image_path, question):
    """Evaluate a single image-question pair and return the response"""
    
    # Add instruction for Answer + Rationale format with stronger emphasis
    # Format: Answer: [answer] Rationale: [rationale]
    # Put format requirement first to make it more prominent
    qs = f"IMPORTANT: You must respond in the following exact format:\n\nAnswer: [your answer]\nRationale: [your reasoning]\n\nQuestion: {question}"
    
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    model_name = get_model_name_from_path(args.model_path)
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            f"[WARNING] the auto inferred conversation mode is {conv_mode}, while `--conv-mode` is {args.conv_mode}, using {args.conv_mode}"
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Load and process image
    image = load_image(image_path)
    image_sizes = [image.size]
    images_tensor = process_images(
        [image],
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    
    # Extract the model's response (remove the prompt part)
    # The response is typically after the last occurrence of the assistant role marker
    if conv.roles[1] in outputs:
        response = outputs.split(conv.roles[1])[-1].strip()
    else:
        response = outputs
    
    # Format response to ensure it follows Answer: ... Rationale: ... format
    if args.ensure_format:
        response = format_response(response)
    
    return response


def process_haloquest_data(args):
    """Process haloquest CSV file and generate Answer + Rationale for each question"""
    
    # Load model once
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )
    
    # Read CSV file
    results = []
    successful_count = 0  # Counter for successfully processed samples
    
    with open(args.csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        total_rows = len(rows)
        
        if args.limit and args.limit > 0:
            print(f"[INFO] Will process up to {args.limit} successful samples (errors not counted)")
        
        for idx, row in enumerate(rows, 1):
            # Check if we've reached the limit of successful samples
            if args.limit and args.limit > 0 and successful_count >= args.limit:
                print(f"\n[INFO] Reached limit of {args.limit} successful samples. Stopping.")
                break
            
            image_name = row['image_name']
            question = row['question']
            image_type = row.get('image type', '')
            hallucination_type = row.get('hallucination type', '')
            split = row.get('split', '')
            
            # Construct local image path
            image_path = os.path.join(args.image_dir, image_name)
            
            if not os.path.exists(image_path):
                print(f"[{idx}/{total_rows}] [WARNING] Image not found: {image_path}, skipping (not counted)")
                results.append({
                    'image_name': image_name,
                    'question': question,
                    'answer_rationale': 'ERROR: Image not found',
                    'image_type': image_type,
                    'hallucination_type': hallucination_type,
                    'split': split
                })
                continue
            
            print(f"[{idx}/{total_rows}] Processing: {image_name} (Success: {successful_count}/{args.limit if args.limit else 'N/A'})")
            print(f"  Question: {question}")
            
            try:
                # Generate Answer + Rationale
                response = eval_model_single(
                    tokenizer, model, image_processor, args, image_path, question
                )
                
                print(f"  Response: {response[:100]}...")  # Print first 100 chars
                print()
                
                results.append({
                    'image_name': image_name,
                    'question': question,
                    'answer_rationale': response,
                    'image_type': image_type,
                    'hallucination_type': hallucination_type,
                    'split': split
                })
                
                # Increment successful count only when processing succeeds
                successful_count += 1
                
            except Exception as e:
                print(f"  [ERROR] Failed to process: {str(e)} (not counted)")
                results.append({
                    'image_name': image_name,
                    'question': question,
                    'answer_rationale': f'ERROR: {str(e)}',
                    'image_type': image_type,
                    'hallucination_type': hallucination_type,
                    'split': split
                })
    
    # Write results to output CSV
    output_file = args.output_file
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['image_name', 'question', 'answer_rationale', 'image_type', 
                     'hallucination_type', 'split']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to: {output_file}")
    print(f"Total rows processed: {len(results)}")
    if args.limit and args.limit > 0:
        print(f"Successful samples: {successful_count}/{args.limit}")
    else:
        # Count successful samples (those without ERROR prefix)
        successful_in_results = sum(1 for r in results if not r['answer_rationale'].startswith('ERROR:'))
        print(f"Successful samples: {successful_in_results}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process haloquest data with LLaVA")
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b",
                       help="Path to the LLaVA model")
    parser.add_argument("--model-base", type=str, default=None,
                       help="Base model path (if different from model-path)")
    parser.add_argument("--csv-file", type=str, required=True,
                       help="Path to haloquest CSV file")
    parser.add_argument("--image-dir", type=str, required=True,
                       help="Directory containing downloaded images")
    parser.add_argument("--output-file", type=str, required=True,
                       help="Output CSV file path")
    parser.add_argument("--conv-mode", type=str, default=None,
                       help="Conversation mode (auto-detected if not specified)")
    parser.add_argument("--temperature", type=float, default=0.2,
                       help="Temperature for generation")
    parser.add_argument("--top-p", type=float, default=None,
                       help="Top-p for generation")
    parser.add_argument("--num-beams", type=int, default=1,
                       help="Number of beams for generation")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                       help="Maximum number of new tokens to generate")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit the number of samples to process (for testing)")
    parser.add_argument("--ensure-format", action="store_true", default=False,
                       help="Post-process responses to ensure Answer: ... Rationale: ... format")
    
    args = parser.parse_args()
    
    process_haloquest_data(args)

