from transformers import T5ForConditionalGeneration, T5Tokenizer


model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

def generate_text(input_text, task="translate English to French"):
    
    input_text = f"{task}: {input_text}"
    
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    
    outputs = model.generate(inputs['input_ids'], max_length=50, num_beams=5, early_stopping=True)
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


input_text = input("Please enter a sentence to translate: ")


output_text = generate_text(input_text, task="translate English to French")


print("Translated Text:")
print(output_text)