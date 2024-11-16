
def simple_sample(model, prompt, num_samples,temperature):
    outputs = []
    for _ in range(num_samples):
        output,log = model.predict(prompt,temperature=temperature,return_full=False)
        outputs.append(output)
    return outputs

