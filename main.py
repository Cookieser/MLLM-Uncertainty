import os
from PIL import Image
from uncertainty.models.mllm_model import MLLMModel
from uncertainty.models.huggingface_models import HuggingfaceModel
# Directory path


# Counter for images found

def transformed_images_address(idx,base_path):
# Iterate through folders
    image_count = 0
    image_paths = []

    # Iterate through folders
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            image_path = os.path.join(folder_path, f"{idx}.png")
            if os.path.exists(image_path):

                image_paths.append(image_path)
                image_count += 1

    # Print the total count
    return image_count,image_paths


base_path = "/work/images/images"
image_count,image_paths = transformed_images_address(1,base_path)
print(image_count)


model = MLLMModel('llava-1.5-7b-hf','default',50);


#question = "Only answer Yes or No: Is this a dog?"
#image_file = image_paths[0]


question = "What is this?A.apple B.ship"
image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
temperature = 0.7


answer,clean_log_likelihoods,last_token_embedding = model.predict(question,image_file,temperature)


print("Clean Answer:", answer)
print("Log-Likelihoods:", clean_log_likelihoods)
print("Last Token Embedding Shape:", last_token_embedding.shape)

p = model.get_p_true(question,image_file)
print(p)
