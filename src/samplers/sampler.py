from .simple_sampler import simple_sample

LOW_TEMPERATURE = 0.1
class Sampler:
    def __init__(self,prompt,model, sample_method,temperature=LOW_TEMPERATURE):
        self.model = model
        self.sample_method =sample_method
        self.outputs = []
        self.prompt = prompt
        self.temperature = temperature

    def sample(self, num_samples):

        if self.sample_method == "simple":
            new_outputs = simple_sample(self.model,self.prompt,num_samples,self.temperature)
            self.outputs.extend(new_outputs)
            return self.outputs
        elif self.sample_method == "similar":
            return self._similar_sample(self.prompt,num_samples)
        else:
            raise ValueError(f"Unsupported sample method: {self.sample_method}")   

    def show_all_result(self):
        if not self.outputs:
            print("No outputs available. Please call `sample` first.")
            return
        print(f"Use the '{self.sample_method}' Method in the temperature {self.temperature}:")
        for i, result in enumerate(self.outputs, start=1):
            print(f"Sample {i}: {result}")  
        