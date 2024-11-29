import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

if __name__ == "__main__":
    model_name = "./Llama-3.2-3B-Instruct-emotion-final"
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", trust_remote_code=True)
    pipe = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device=0,
    )
    system_prompt = input("Enter a system prompt for text generation: ")
    messages = [
            {"role": "system", "content": system_prompt}
        ]
    while(True):
        user_prompt = input("Enter a prompt for text generation: ")
        messages.append({"role": "user", "content": user_prompt})
        generate_text = pipe(
            messages,
            max_new_tokens=256,
        )
        messages.append(generate_text[0]["generated_text"][-1])
        print(messages[-1]["content"])
        print("\n")