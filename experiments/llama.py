from transformers import AutoTokenizer, LlamaForCausalLM

model = LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-70b-chat')
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-70b-chat')


while True:
    prompt = input("User : ")

    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate
    generate_ids = model.generate(inputs.input_ids, max_length=30)
    output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(f'System : {output}')