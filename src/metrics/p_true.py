#  Language Models (Mostly) Know What They Know

#  Saurav Kadavath et al.,21 Nov 2022 
#  <https://arxiv.org/abs/2207.05221>
def calculate_p_true(model, question, choices,answer,truth = "A"):
    judge_prompt = f"""Question: {question}
Here are some brainstormed ideas: {choices}
Possible Answer: {answer}
Is the possible answer:
(A) True
(B) False
The possible answer is:"""

    return  model.get_p_true(judge_prompt,truth)
