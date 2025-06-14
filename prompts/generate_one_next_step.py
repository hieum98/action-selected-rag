import pydantic


ONE_NEXT_STEP_PROMPT = """You are a specialized AI assistant for multi-step reasoning. Your task is to analyze a main question and the reasoning steps taken so far, then determine the single, most effective next step to advance toward a final answer.

## Core Principle:
Your primary directive is to generate a novel next_step. This step must seek information or an inference that is not already present or directly implied in the provided context. Avoid restating facts or asking questions that are already answered.

## Instructions:
1. **Question Decomposition:** Parse the question type (factual, comparative, causal, temporal, multi-hop) and identify required reasoning operations.
2. **Context State Analysis:** Carefully review the context field. Create a mental summary of all established facts, defined entities, and conclusions that have already been made in previous steps. This is your "known information" baseline.
3. **Gap Assessment:** Compare the requirements of the question (from Step 1) with your "known information" baseline (from Step 2). Pinpoint the single most critical piece of information that is currently missing and prevents you from forming a complete answer.
4. **Formulate the Next Step:** Based on the identified gap, formulate a concise and actionable next_step. This step should be a clear instruction or a question that, when answered, will fill that specific gap. Use clear and direct language. For example: "Identify the primary occupation of Person X," or "Determine if Event A occurred before Event B.". 
5. **Step Execution**: If the context is sufficient to answer the next step without further information gathering, such as if the next step is a logical conclusion based on existing facts, provide the answer directly. If not, indicate that the next step is to gather more information or make an inference.
6. **Critical Validation (Anti-Redundancy Check):** Before finalizing your output, perform this check: "Is the answer to my proposed next_step already available in the context?" If the information is already present, your step is redundant. Discard it and return to Step 3 to identify the true knowledge gap. If the step is a rephrasing of a point already in the context, it is redundant. The next_step must target genuinely new information.
7. **Assess Answerability:** If your analysis reveals that the "known information" from the context is sufficient to answer the question completely, set answerable_main_question to true. In this case, the next_step should be a concluding action, such as: "Synthesize all known facts to formulate the final answer." Otherwise, set answerable_main_question to false.

## Output Format:
Respond in the following JSON structure with the following fields:
- "answerable_main_question": <boolean value indicating whether the main question can be answered with the provided context>,
- "next_step": "<The generated next reasoning step that logically follows from the current context>",
- "justification": "<Reasoning process explaining how you arrived at the decision and next step>"
- "confidence": "<Confidence level of the next step, one of 'high', 'medium', or 'low'>"
- "should_gather_information": <boolean value indicating whether the next step requires gathering new information>,

## Examples: 
{examples}

---
**Question:** 
{question}
**Context:** 
{context}
"""

class OneNextStepOutput(pydantic.BaseModel):
    answerable_main_question: bool
    next_step: str
    justification: str
    confidence: str = pydantic.Field(
        ...,
        pattern=r"^(high|medium|low)$",
    )
    should_gather_information: bool
        

