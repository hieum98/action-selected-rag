from enum import Enum
from typing import Union, Literal
import pydantic


EVALUATE_ANSWER_PROMPT = """You are an expert assistant specializing in evaluating the quality of answers to questions. For each task, you will receive a question, a correct answer, and a predicted answer to evaluate. Your goal is to compare the predicted answer with the correct answer and determine whether the predicted answer is matched with the correct answer.  

Instructions:
1. **Question Analysis:** Carefully read and understand the question. Identify key components and clarify what is being asked.
2. **Answer Comparison:** Compare the predicted answer with the correct answer. Evaluate the following aspects:
- The predicted answer convey the same meaning as the correct answer, not just exact matching.
- The predicted answer cover all necessary aspects of the correct answer
- The predicted answer is considered correct if it is matched with ANY ONE of the correct answers.
3. **Output Format:** Respond in the following JSON structure with the following fields:
- "decision": "<boolean value indicating whether the predicted answer is matched with the correct answer, 'true' if it is matched, 'false' otherwise>",
- "reasoning": "<Step-by-step reasoning process explaining how you arrived at the evaluation result>"

Here are some examples: {examples}

Now, please evaluate the following question and answers:
Question: {question}
Correct Answer: {correct_answer}
Predicted Answer: {predicted_answer}
"""

class EvaluateAnswerOutput(pydantic.BaseModel):
    decision: bool 
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


EVALUATE_SAME_QUESTION_PROMPT = """You are an expert assistant specializing in evaluating whether two questions are the same or not. For each task, you will receive two questions. Your goal is to determine whether the two questions are the same or not.

Instructions:
1. **Question Analysis:** Carefully read and understand both questions. Identify key components and clarify what is being asked in each question.
2. **Comparison:** Compare the two questions to determine if they are asking the same thing. Note that two questions are considered the same if they convey the same meaning, and the answer to one question can be used to answer the other question.
3. **Output Format:** Respond in the following JSON structure with the following fields:
- "decision": "<boolean value indicating whether the two questions are the same or not, 'true' if they are the same, 'false' otherwise>",
- "reasoning": "<Step-by-step reasoning process explaining how you arrived at the evaluation result>"

Here are some examples: {examples}

Now, please evaluate the following two questions:
Question 1: {question_1}
Question 2: {question_2}
"""

class EvaluateSameQuestionOutput(pydantic.BaseModel):
    decision: bool
    reasoning: str


EVALUATE_ANSWER_GIVEN_CONTEXT_PROMPT = """You are an expert assistant specializing in evaluating the quality of answers to questions given a context. For each task, you will receive a question, a context, and a predicted answer to evaluate. Your goal is to evaluate whether the predicted answer is aligned, in conflict or cannot be determined based on the context provided.

Instructions:
1. **Question Analysis:** Carefully read and understand the question. Identify key components and clarify what is being asked.
2. **Context Analysis:** Read the context provided and identify the key information that is relevant to the question.
3. **Answer Evaluation:** Compare the predicted answer with the context. Evaluate the following aspects:
- The predicted answer is aligned with the context if it is supported by the information in the context.
- The predicted answer is in conflict with the context if it contradicts the information in the context.
- The predicted answer cannot be determined based on the context if the context does not provide enough information to evaluate the answer.
4. **Output Format:** Respond in the following JSON structure with the following fields:
- "decision": "<'aligned' if the predicted answer is aligned with the context, 'in_conflict' if it is in conflict with the context, 'cannot_be_determined' if it cannot be determined based on the context>",
- "reasoning": "<Step-by-step reasoning process explaining how you arrived at the evaluation result>"

Here are some examples: {examples}

Now, please evaluate the following question and answer given the context:
Question: {question}
Context: {context}
Predicted Answer: {predicted_answer}
"""

class EvaluateAnswerGivenContextOutput(pydantic.BaseModel):
    decision: str = pydantic.Field(
        ..., 
        pattern=r"^(aligned|in_conflict|cannot_be_determined)$",
    )
    reasoning: str
        

