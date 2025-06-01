import pydantic


EVALUATE_ANSWER_PROMPT = """You are an expert assistant specializing in evaluating the quality of answers to questions. For each task, you will receive a question, a correct answer, and a predicted answer to evaluate. Your goal is to compare the predicted answer with the correct answer and determine whether the predicted answer is matched with the correct answer.  

Instructions:
1. **Question Analysis:** Carefully read and understand the question. Identify key components and clarify what is being asked.
2. **Answer Comparison:** Compare the predicted answer with the correct answer. Evaluate the following aspects:
- The predicted answer convey the same meaning as the correct answer, not just exact matching.
- The predicted answer cover all necessary aspects of the correct answer
- The predicted answer is considered correct if it is matched with ANY ONE of the correct answers.
3. **Output Format:** Respond in the following JSON structure with the following fields:
- "result": "<'matched' if the predicted answer is matched with the correct answer, 'not_matched' otherwise>",
- "reasoning": "<Step-by-step reasoning process explaining how you arrived at the evaluation result>"

Here are some examples: {examples}

Now, please evaluate the following question and answers:
Question: {question}
Correct Answer: {correct_answer}
Predicted Answer: {predicted_answer}
"""

class EvaluateAnswerOutput(pydantic.BaseModel):
    result: str 
    reasoning: str  

