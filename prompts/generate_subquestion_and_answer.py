import pydantic


SUBQUESTION_AND_ANSWER_PROMPT = """You are an expert assistant specializing in multi-hop question answering and reasoning decomposition. Your task is to generate a subquestion that logically follows from the main question and provide a direct answer to that subquestion, accompanied by transparent, step-by-step reasoning.

## Core Objectives:
1. **Subquestion Generation:** Analyze the main question and context to identify gaps in reasoning or knowledge, then generate a strategic subquestion that advances the reasoning process.
2. **Direct Answer Provision:** Provide a direct, evidence-based answer to the generated subquestion, ensuring clarity and completeness.

## Decision Framework:
**Subquestion Quality Criteria:**
- **Specificity:** Targets a well-defined knowledge gap
- **Independence:** Can be answered without requiring the main question's answer
- **Relevance:** Directly contributes to solving the main question
- **Atomicity:** Focuses on a single reasoning step or concept
**Answer Quality Criteria:**
- **Relevance:** Directly addresses the subquestion without unnecessary information
- **Accuracy:** Based on the most reliable and relevant information available
- **Clarity:** Clearly articulated without ambiguity
- **Completeness:** Covers all aspects of the subquestion as required

## Instructions:
1. **Parse Question Components:** Identify the question type (factual, comparative, causal, temporal), key entities, and required reasoning steps.
2. **Context Gap Analysis:** Map provided context against question requirements. Identify missing entities, incomplete relationships, or logical gaps.
3. **Subquestion Strategy:** Generate a subquestion using these guidelines:
   - **For factual gaps:** "What is [missing entity/definition]?"
   - **For relationship gaps:** "How does [entity A] relate to [entity B]?"
   - **For causal gaps:** "What causes [effect] in [context]?"
   - **For temporal gaps:** "When did [event] occur relative to [reference]?"
4. **Direct Answer Generation:**
   - Provide a concise, direct answer to the generated subquestion based on the provided context.
   - Document your reasoning process in clear, logical steps, showing how you arrived at the answer.
5. **Validation:** Ensure your subquestion is answerable independently and advances toward the main question's resolution. The answer should be based on the context provided or general knowledge if necessary.

## Output Format:
Respond in the following JSON structure with the following fields:
- "subquestion": "<The generated subquestion that logically follows from the main question>",
- "answer": "<Direct, qualified answer to the generated subquestion>",
- "reasoning": "<Step-by-step thought process, including context analysis, knowledge gaps identified, and logical steps taken>",
- "confidence": "<High/Medium/Low based on evidence quality and completeness>"

## Examples: 
{examples}

---
**Question:** 
{question}
**Context:** 
{context}
"""

class SubquestionAndAnswerOutput(pydantic.BaseModel):
    subquestion: str
    answer: str
    reasoning: str
    confidence: str = pydantic.Field(
         ...,
         pattern=r"^(High|Medium|Low)$",
    )


GENERATE_SUBQUESTION_PROMPT = """You are an expert assistant specializing in multi-hop question answering and reasoning decomposition. Your task is to analyze whether a main question can be answered with the provided context, and if not, generate a strategic subquestion that advances the reasoning process.

## Core Objectives:
1. **Assess Context Sufficiency:** Determine if the provided context contains sufficient information to answer the main question completely and accurately.
2. **Strategic Subquestion Generation:** When context is insufficient, create a subquestion that bridges knowledge gaps and follows logical reasoning chains.

## Decision Framework:
**Context Sufficiency Criteria:**
- All key entities and relationships mentioned in the question are covered
- Temporal/causal dependencies are satisfied
- No critical reasoning steps are missing
- Answer can be derived with high confidence
**Subquestion Quality Criteria:**
- **Specificity:** Targets a well-defined knowledge gap
- **Independence:** Can be answered without requiring the main question's answer
- **Relevance:** Directly contributes to solving the main question
- **Atomicity:** Focuses on a single reasoning step or concept

## Instructions:
1. **Parse Question Components:** Identify the question type (factual, comparative, causal, temporal), key entities, and required reasoning steps.
2. **Context Gap Analysis:** Map provided context against question requirements. Identify missing entities, incomplete relationships, or logical gaps.
3. **Answerability Assessment:** Apply sufficiency criteria to determine if context enables a complete, confident answer.
4. **Subquestion Strategy:** If insufficient, generate a subquestion using these guidelines:
   - **For factual gaps:** "What is [missing entity/definition]?"
   - **For relationship gaps:** "How does [entity A] relate to [entity B]?"
   - **For causal gaps:** "What causes [effect] in [context]?"
   - **For temporal gaps:** "When did [event] occur relative to [reference]?"
5. **Validation:** Ensure your subquestion is answerable independently and advances toward the main question's resolution.

## Output Format:
Respond in the following JSON structure with the following fields:
- "answerable_main_question": <boolean value indicating whether the main question can be answered with the provided context>,
- "subquestion": "<The generated subquestion that logically follows from the main question>",
- "reasoning": "<Reasoning process explaining how you arrived at the decision and subquestion>",
- "gap_type": "<Type of reasoning gap identified, one of 'factual', 'relational', 'causal', 'temporal', 'logical', or 'null'>"

## Examples: 
{examples}

---
**Question:** 
{question}
**Context:** 
{context}
"""

class SubquestionOutput(pydantic.BaseModel):
    answerable_main_question: bool
    subquestion: str
    reasoning: str
    gap_type: str = pydantic.Field(
        ...,
        pattern=r"^(factual|relational|causal|temporal|logical|null)$",
    )
        
