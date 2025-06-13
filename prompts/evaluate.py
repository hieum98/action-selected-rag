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
- "confidence": "<Confidence level of the evaluation, e.g., 'high', 'medium', 'low'>"

Here are some examples: {examples}

Now, please evaluate the following question and answers:
Question: {question}
Correct Answer: {correct_answer}
Predicted Answer: {predicted_answer}
"""

class EvaluateAnswerOutput(pydantic.BaseModel):
    decision: bool 
    reasoning: str  
    confidence: Literal['high', 'medium', 'low'] = pydantic.Field(
        ...,
        pattern=r"^(high|medium|low)$",
    )


EVALUATE_RETRIEVED_DOCUMENT_PROMPT = """You are an expert assistant specializing in evaluating the relevance of a retrieved document to a given question and the current step's objective/ subquestions. For each task, you will receive a question, the current step's objective, and a document to evaluate. Your goal is to determine whether the document is relevant to the question or the current step's objective.

Instructions:
1. **Question and Objective Analysis:** Carefully read and understand the question and the current step's objective. Identify key components and clarify what is being asked.
2. **Document Relevance Evaluation:** Analyze the document to determine whether it provides relevant information to answer the question or the current's step's objective. Note that the relevance document may not directly answer the question or the current step's objective, but it should provide useful information that can help answer the question or the current step's objective.
3. **Thought Process**: Provide a brief analysis for each document, considering both the answer content and the current objectives.
4. **Comprehensive Information Extraction:** If the document is relevant, extract ALL useful information using the following structured approach:
   - **Direct Answers:** Information that directly addresses the question or objective
   - **Supporting Evidence:** Context, background, or related facts that strengthen understanding
   - **Partial Information:** Incomplete but relevant details that could be combined with other sources
   - **Contradictory Information:** Any information that challenges assumptions or provides alternative perspectives
5. **Information Formatting:** Present extracted information as a coherent paragraph with bullet points. For each point, include:
   - The key information summary
   - The exact original text in quotation marks as evidence
   - Brief explanation of relevance and confidence level
6. **Output Format:** Respond in the following JSON structure with the following fields:
- "decision": "<'relevant' if the document is relevant to the question and the current step's objective, 'not_relevant' otherwise>",
- "reasoning": "<Step-by-step reasoning process explaining how you arrived at the evaluation result>"
- "extracted_information": "<A paragraph describing the relevant information with bullet points. Each bullet point should contain: the information summary, original quoted text from the document, and explanation of how it relates to the question/objective. If not relevant, leave as empty string>"

**Key Requirements:**
- Always include exact quotes from the original document as proof
- Structure information as bullet points within a flowing paragraph
- Explain the relevance and confidence for each extracted piece
- Extract information even if it only partially addresses the question

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


EVALUATE_ANSWER_GIVEN_CONTEXT_PROMPT = """You are an expert assistant specializing in rigorous answer evaluation for question answering systems. For each task, you will receive a question, context, and a predicted answer. Your goal is to systematically determine the relationship between the predicted answer and the supporting context through comprehensive verification.

## Evaluation Framework:
**Assessment Categories:**
- **aligned:** The predicted answer is fully supported by explicit information in the context
- **in_conflict:** The predicted answer directly contradicts information stated in the context  
- **cannot_be_determined:** The context lacks sufficient information to verify or refute the answer

**Verification Criteria:**
- **Factual Accuracy:** All factual claims in the answer must be verifiable against the context
- **Logical Consistency:** The answer must follow logically from the available information
- **Completeness Alignment:** The answer scope should match what the context can support
- **Inference Validity:** Any inferences must be reasonable given the contextual evidence

## Instructions:
1. **Question Decomposition:** Parse the question to identify required information types, scope, and expected answer format.
2. **Context Mapping:** Extract all relevant facts, relationships, and evidence from the context that relate to the question.
3. **Answer Analysis:** Break down the predicted answer into component claims and assertions.
4. **Systematic Verification:** For each answer component:
   - Check direct factual support in context
   - Verify logical consistency with contextual information
   - Assess whether inferences are justified by available evidence
5. **Decision Synthesis:** Apply evaluation criteria to determine overall alignment status.

## Output Format:
Respond in the following JSON structure with the following fields:
- "decision": "<'aligned', 'in_conflict', or 'cannot_be_determined'>",
- "reasoning": "<Detailed reasoning process explaining how you arrived at the evaluation result, including specific evidence from the context that supports your decision>"

## Examples: 
{examples}

---
**Question:**
{question}
**Context:**
{context}
**Predicted Answer:**
{predicted_answer}
"""

class EvaluateAnswerGivenContextOutput(pydantic.BaseModel):
    decision: str = pydantic.Field(
        ..., 
        pattern=r"^(aligned|in_conflict|cannot_be_determined)$",
    )
    reasoning: str


EVALUATE_REASONING_PROMPT = """You are an expert assistant specializing in evaluating the quality of reasoning processes. You will be given: Original Question: The question the reasoning path attempts to answer; Reasoning Path: The sequence of steps, arguments, or inferences presented as the solution or explanation.; Correct Answer (Optional): The known correct answer to the Original Question. If not provided, the evaluation will focus solely on the intrinsic quality of the reasoning.
Please analyze the provided Reasoning Path based on the following criteria. Structure your evaluation to address each point clearly, providing specific examples or references to the steps in the Reasoning Path where appropriate.

## Instructions:
1. **Step-by-Step Analysis:** For each distinct step or component in the Reasoning Path:
    - Logical Validity: Does the conclusion or assertion of this step logically follow from the preceding steps, given premises, or provided context? Identify any logical fallacies or gaps in inference.
    - Factual Accuracy & Grounding: Are the claims, data, evidence, or premises introduced or utilized in this step factually correct? If context or documents are provided, is the information accurately drawn from and consistent with them? Note any inaccuracies or unsupported claims.
    - Clarity & Precision: Is the language used in this step clear, precise, and unambiguous? Are there any terms or statements that are vague or could lead to misinterpretation? 
    - Relevance: Does this step directly and meaningfully contribute to addressing the Original Question and reaching the final conclusion?
2. **Overall Path Evaluation:** You will then assess the Reasoning Path as a whole, considering the following criteria:
    - Coherence: Does the entire Reasoning Path demonstrate a logical and understandable flow? Do the steps connect smoothly and build upon each other in a cohesive manner? 
    - Completeness & Sufficiency: Does the path include all necessary intermediate steps, information, and considerations required to logically bridge the gap from the Original Question to the final conclusion? Are there any critical omissions? Conversely, are there any redundant or superfluous steps that do not add value
    - Consistency: Are there any internal contradictions or inconsistencies between different parts of the Reasoning Path?
3. **Conclusion Assessment:** Based on whether a Correct Answer is provided or not, you will evaluate the final conclusion of the Reasoning Path:
    - If a Correct Answer is provided: Does the Reasoning Path ultimately arrive at the Correct Answer? If the path's conclusion is incorrect, pinpoint the earliest step(s) where the error (logical, factual, calculational, misinterpretation, etc.) occurs that leads to the deviation. If the path's conclusion matches the Correct Answer, critically assess whether the reasoning process itself is sound, complete, and free of significant flaws. (It is possible to reach a correct answer through flawed reasoning).
    - If no Correct Answer is provided (focus on intrinsic quality):
        - Based solely on the structure, content, and evidence within the Reasoning Path, how convincing and well-supported is the stated conclusion?  
        - What are the most significant strengths of the reasoning in supporting its conclusion?
        - What are the most critical weaknesses or vulnerabilities in the reasoning that might undermine its conclusion?
        - Overall Summary and Recommendations (Optional but Encouraged):

## Output Format:
Respond in the following JSON structure with the following fields:
- "step_quality": "<The quality of reasoning path when evaluating each step, i.e., inter-step. This should be one of 'excellent', 'good', 'fair', 'poor', or 'very_poor'>",
- "step_quality_details": "<Detailed evaluation of each step in the reasoning path, addressing logical validity, factual accuracy, clarity, and relevance. Provide specific examples or references to the steps in the Reasoning Path where appropriate>",
- "overall_quality": "<The overall quality of the reasoning path, i.e., intra-step. This should be one of 'excellent', 'good', 'fair', 'poor', or 'very_poor'>",
- "overall_quality_details": "<Overall evaluation of the reasoning path, addressing coherence, completeness, consistency, and the final conclusion. Provide specific examples or references to the steps in the Reasoning Path where appropriate>",
- "conclusion_quality": "<The quality of the final conclusion of the reasoning path, i.e., final step. This should be one of 'excellent', 'good', 'fair', 'poor', or 'very_poor'>",
- "conclusion_quality_details": "<Evaluation of the final conclusion, addressing correctness (if a correct answer is provided), soundness, and any critical strengths or weaknesses in the reasoning process. Provide specific examples or references to the steps in the Reasoning Path where appropriate>"

Here are some examples: {examples}

Now, please evaluate the following reasoning process:
Original Question: {original_question}
Reasoning Path: {reasoning_path}
Correct Answer (Optional): {correct_answer}
"""

class EvaluateReasoningOutput(pydantic.BaseModel):
    step_quality: Literal['excellent', 'good', 'fair', 'poor', 'very_poor'] = pydantic.Field(
        ...,
        pattern=r"^(excellent|good|fair|poor|very_poor)$",
    )
    step_quality_details: str
    overall_quality: Literal['excellent', 'good', 'fair', 'poor', 'very_poor'] = pydantic.Field(
        ...,
        pattern=r"^(excellent|good|fair|poor|very_poor)$",
    )
    overall_quality_details: str
    conclusion_quality: Literal['excellent', 'good', 'fair', 'poor', 'very_poor'] = pydantic.Field(
        ...,
        pattern=r"^(excellent|good|fair|poor|very_poor)$",
    )
    conclusion_quality_details: str
   
        

