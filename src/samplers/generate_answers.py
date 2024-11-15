from models.huggingface_models import *
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-hf", help="Huggingface model name")
    parser.add_argument('--stop_sequences', type=str, default="default", help="Stop sequences for the model")
    parser.add_argument('--max_new_tokens', type=int, default=50, help="Maximum number of new tokens")
    parser.add_argument('--temperature', type=float, default=0.7, help="Sampling temperature")
    parser.add_argument('--return_full', action='store_true', help="Return full output text")
    return parser.parse_args()





def get_answers(args):
    """Generate answers based on input questions."""
    huggingface_model = HuggingfaceModel(
        model_name=args.model_name,
        stop_sequences=args.stop_sequences,
        max_new_tokens=args.max_new_tokens
    )

    user_tag = "USER:"
    assistant_tag = "ASSISTANT:"
    all_questions = ["What is the capital of France?"]


    template_str = '{user_tag}{question}{assistant_tag}'
    input = [template_str.format(user_tag=user_tag,question = q,assistant_tag=assistant_tag) for q in all_questions]
    print(input)    
    input_text = input[0]
    print(args.return_full)

    try:
        output_text = huggingface_model.predict(input_text, temperature=args.temperature, return_full=args.return_full)
        print("Generated Text >\n", output_text)
    except Exception as e:
        print("Error during model prediction:", str(e))




if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    # Parse arguments
    args = get_args()

    # Call get_answers with parsed arguments
    get_answers(args)
