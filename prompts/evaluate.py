import pydantic


EVALUATE_ANSWER_PROMPT = """You are an expert assistant specializing in evaluating the quality of answers to questions. For each task, you will receive a question, a correct answer, and a predicted answer to evaluate. Your goal is to compare the predicted answer with the correct answer and determine whether the predicted answer is matched with the correct answer.  

Instructions:
1. **Question Analysis:** Carefully read and understand the question. Identify key components and clarify what is being asked.
2. **Answer Comparison:** Compare the predicted answer with the correct answer. Evaluate the following aspects:
- The predicted answer convey the same meaning as the correct answer, not just exact matching.
- The predicted answer cover all necessary aspects of the correct answer
- The predicted answer is considered correct if it is matched with ANY ONE of the correct answers.
3. **Output Format:** Respond in the following JSON structure with the following fields:
- "decision": "<'matched' if the predicted answer is matched with the correct answer, 'not_matched' otherwise>",
- "reasoning": "<Step-by-step reasoning process explaining how you arrived at the evaluation result>"

Here are some examples: {examples}

Now, please evaluate the following question and answers:
Question: {question}
Correct Answer: {correct_answer}
Predicted Answer: {predicted_answer}
"""

class EvaluateAnswerOutput(pydantic.BaseModel):
    decision: str 
    reasoning: str  



EVALUATE_RETRIEVED_DOCUMENT_PROMPT = """You are an expert assistant specializing in evaluating the relevance of a retrieved document to a given question and the current step's objective/ subquestions. For each task, you will receive a question, the current step's objective, and a document to evaluate. Your goal is to determine whether the document is relevant to the question or the current step's objective.

Instructions:
1. **Question and Objective Analysis:** Carefully read and understand the question and the current step's objective. Identify key components and clarify what is being asked.
2. **Document Relevance Evaluation:** Analyze the document to determine whether it provides relevant information to answer the question or the current's step's objective. Note that the relevance document may not directly answer the question or the current step's objective, but it should provide useful information that can help answer the question or the current step's objective.
3. **Thought Process**: Provide a brief analysis for each document, considering both the answer content and the current objectives.
4. **Information Extraction:** If the document is relevant, extract the information that can help answer the question or fulfill the current step's objective.
5. **Output Format:** Respond in the following JSON structure with the following fields:
- "decision": "<'relevant' if the document is relevant to the question and the current step's objective, 'not_relevant' otherwise>",
- "reasoning": "<Step-by-step reasoning process explaining how you arrived at the evaluation result>"
- "extracted_information": "<Extracted information from the document that is relevant to the question and the current step's objective or an empty string if the document is not relevant>"

Here are some examples: {examples}

Now, please evaluate the following document given the question and the current step's objective:
Question: {question}
Current Step's Objective: {current_step_objective}
Document: {document}
"""

class EvaluateRetrievedDocumentOutput(pydantic.BaseModel):
    decision: str
    reasoning: str
    extracted_information: str
