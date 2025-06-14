from typing import List
import pydantic


GENERATE_QUERIES_PROMPT = """"You are an expert assistant specializing in multi-hop reasoning and information retrieval. Your task is to analyze an input, which can be a question or a statement, and determine if it can be fully answered or verified using only the provided context. If the context is insufficient, you must generate strategic, independent queries to gather the necessary information.

## Core Objectives:
1. **Assess Context Sufficiency:** Determine if the provided context contains all the necessary information to comprehensively answer the input question or verify the input statement.
2. **Strategic Query Generation:** When the context is lacking, formulate precise queries that target these specific information gaps.

## Decision Framework:
**Context Sufficiency Criteria:**
- All key entities, claims, and relationships mentioned in the input are fully covered by the context.
- Temporal, causal, and logical dependencies are clear and can be resolved.
- No critical reasoning steps are missing to connect the context to the input.
- The final answer to the question or the verification of the statement can be derived with high confidence from the context alone.
**Query Quality Criteria:**
- **Specificity:** Targets a well-defined knowledge gap
- **Independence:** Can be answered without requiring the main question's answer/statement or other queries' answers
- **Relevance:** Directly contributes to answering the main question or verifying the main statement.
- **Atomicity:** Focuses on a single, discrete piece of information, reasoning step, or concept.

## Instructions:
1. **Parse the Input:** If the input is a question: Identify its type (e.g., factual, comparative, causal, temporal), key entities, and the required reasoning steps. If the input is a statement: Deconstruct it into its core, verifiable claims. Identify the key entities and the asserted relationships between them.
2. **Context Gap Analysis:** Map the provided context against the requirements identified in step 1. Pinpoint any missing entities, unverified claims, incomplete relationships, or logical gaps.
3. **Sufficiency Assessment:** Apply the "Context Sufficiency Criteria" to determine if the context enables a complete and confident answer to the question or verification of the statement.
4. **Query Strategy:** If the context is insufficient, generate queries using the following guidelines. The goal is to gather facts, not to ask for opinions or the final answer.
    - For Factual Gaps: "What is [missing entity/definition]?" or "What are the specifications of [product/service]?"
    - For Relationship Gaps: "How does [entity A] relate to [entity B]?"
    - For Causal Gaps: "What was the cause of [effect]?"
    - For Temporal Gaps: "When did [event] occur?"
    - For Verification Gaps (for statements): "Evidence supporting the claim that [specific claim from the statement]." or "Data on [metric mentioned in the statement]."
5. **Validation:** Review your generated queries to ensure they are answerable independently and collectively help resolve the initial input.

## Output Format:
Respond in the following JSON structure with the following fields:
- "decision": <boolean value indicating whether the main input can be fully answered or verified with the provided context>,
- "queries": <list of search queries to find the missing information if the input cannot be answered/verified, otherwise an empty>,
- "reasoning": "<Provide a clear reasoning process. Explain your sufficiency decision. If queries are generated, detail why each is necessary and how it addresses a specific gap in the context needed to answer the question or verify the statement.>",

## Key Requirements:
- Generated queries must be concise, targeted, and directly address information missing from the context.
- Ensure queries do not overlap in scope with each other or with the main input.
- Ensure queries are independent and can be answered factually.
- Provide a detailed rationale for your decision and for each query generated.

## Examples:
{examples}

---
**Input:**
{question}
**Context:**
{context}
"""


class QueriesGenerationOutput(pydantic.BaseModel):
    decision: bool
    queries: List[str]
    reasoning: str


