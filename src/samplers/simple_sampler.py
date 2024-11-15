def simple_sample(model, prompt, num_samples):
    outputs = []
    for _ in range(num_samples):
        output,log = model.predict(prompt)
        outputs.append(output)
    return outputs