############################################
# LLM INFERENCE
############################################

def explain_code(question, code, tokenizer, model):
    messages = [
        {
            "role": "user",
            "content": (
                "You are an expert C programmer and software analyst.\n\n"
                f"User question: {question}\n\n"
                "Relevant C code snippet:\n"
                f"{code.strip()}\n\n"
                "Provide a clear and concise explanation of what this code does "
                "and how it relates to the question above."
            ),
        }
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.2,
        do_sample=True,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode only newly generated tokens — skip the echoed prompt
    input_len = inputs["input_ids"].shape[-1]
    new_tokens = outputs[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
