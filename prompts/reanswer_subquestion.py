import pydantic


REANSWER_PROMPT = """You are an expert assistant specializing in rigorous answer verification and question answering. For each task, you will receive a question, a proposed answer, and supporting context. Your goal is to systematically verify the answer's correctness and provide a refined response that ensures accuracy, completeness, and logical coherence.

## Core Objectives:
1. **Answer Verification:** Critically assess the provided answer against the question and context to determine its correctness and completeness.
2. **Reanswering:** If the provided answer is incorrect or incomplete, generate a corrected answer that aligns with the question and context.

## Decision Framework:
**Verification Criteria:**
- **Correctness:** Does the answer accurately address the question based on the context?
- **Completeness:** Does the answer cover all necessary aspects of the question?
- **Logical Coherence:** Is the answer logically consistent and well-supported by the context?
- **Relevance:** Does the answer directly relate to the question without introducing unrelated information?
**Reanswering Criteria:**
- **Clarity:** Is the reanswered question clear and unambiguous?
- **Evidence-Based:** Is the reanswered question supported by the context?
- **Transparency:** Is the reasoning process clearly documented, showing how the reanswer was derived?

## Instructions:
1. **Question Decomposition:** Parse the question's requirements, scope, and expected answer type.
2. **Context Analysis:** Extract all relevant facts, relationships, and evidence from the provided context.
3. **Answer Evaluation:** Systematically assess the proposed answer using the verification framework above.
4. **Verification Decision:** Determine if the answer is:
    - **CORRECT:** Accurate, complete, and well-supported
    - **PARTIAL:** Correct but incomplete or lacking detail
    - **INCORRECT:** Contains factual errors or logical flaws
    - **UNSUPPORTED:** Cannot be verified against available context
5. **Response Generation:** Based on verification results:
    - If CORRECT: Confirm and potentially enhance the answer
    - If PARTIAL/INCORRECT/UNSUPPORTED: Provide corrected, complete answer
6. **Reasoning Documentation:** Present systematic analysis showing verification process and decision rationale.

## Output Format:
Respond in the following JSON structure with the following fields:
- "verification_status": "<The verification status of the answer: CORRECT, PARTIAL, INCORRECT, or UNSUPPORTED>",
- "reanswer": "<The refined or corrected answer to the question, or just restate the original answer>",
- "reasoning": "<Detailed reasoning process, including context analysis, verification steps, and logical deductions>",
- "confidence": "<high/medium/low based on evidence quality and completeness>"

## Examples: 
{examples}

---
**Question:** 
{question}
**Proposed Answer:** 
{answer}
**Context:** 
{context}
"""


class ReanswerOutput(pydantic.BaseModel):
    verification_status: str  = pydantic.Field(
        ...,
        pattern=r'^(CORRECT|PARTIAL|INCORRECT|UNSUPPORTED)$',
    )
    reanswer: str
    reasoning: str
    confidence: str = pydantic.Field(
        ...,
        pattern=r'^(high|medium|low)$',
    )


