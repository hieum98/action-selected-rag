from typing import List
import pydantic


GENERATE_QUERIES_PROMPT = """"You are an expert assistant specializing in multi-hop question answering and reasoning decomposition. Your task is to analyze whether a main question can be answered with the provided context, and if not, generate strategic queries such that by searching for these queries, it will help to complement the context to answer the main question.

## Core Objectives:
1. **Assess Context Sufficiency:** Determine if the provided context contains sufficient information to answer the main question completely and accurately.
2. **Strategic Query Generation:** When context is insufficient, create queries that bridge knowledge gaps.

## Decision Framework:
**Context Sufficiency Criteria:**
- All key entities and relationships mentioned in the question are covered
- Temporal/causal dependencies are satisfied
- No critical reasoning steps are missing
- Answer can be derived with high confidence
**Query Quality Criteria:**
- **Specificity:** Targets a well-defined knowledge gap
- **Independence:** Can be answered without requiring the main question's answer
- **Relevance:** Directly contributes to solving the main question
- **Atomicity:** Focuses on a single reasoning step or concept

## Instructions:
1. **Parse Question Components:** Identify the question type (factual, comparative, causal, temporal), key entities, and required reasoning steps.
2. **Context Gap Analysis:** Map provided context against question requirements. Identify missing entities, incomplete relationships, or logical gaps.
3. **Answerability Assessment:** Apply sufficiency criteria to determine if context enables a complete, confident answer.
4. **Query Strategy:** Generate queries using these guidelines:
    - **For factual gaps:** "What is [missing entity/definition]?"
    - **For relationship gaps:** "How does [entity A] relate to [entity B]?"
    - **For causal gaps:** "What causes [effect] in [context]?"
    - **For temporal gaps:** "When did [event] occur relative to [reference]?"
5. **Validation:** Ensure your queries are answerable independently and advance toward the main question's resolution.

## Output Format:
Respond in the following JSON structure with the following fields:
- "answerable_main_question": <boolean value indicating whether the main question can be answered with the provided context>,
- "queries": <list of queries that can be used to search for information to complement the context if the main question cannot be answered with the provided context, otherwise an empty list>,
- "reasoning": "<Reasoning process explaining how you arrived at the decision and queries. For each query, explain how it addresses a specific knowledge gap or reasoning step.>",

## Key Requirements:
- Generated queries must be clear, concise, and directly related to the main question.
- Ensure queries are NOT overlapping with the main question or each other.
- Ensure queries are independent and can be answered without needing the main question or other queries's answers.
- Ensuse that the generated queries only focus on the the information that is missing from the context.
- Provide a detailed reasoning for each query, explaining its relevance and how it contributes to answering the main question.

## Examples:
{examples}

---
**Question:**
{question}
**Context:**
{context}
"""


class QueriesGenerationOutput(pydantic.BaseModel):
    answerable_main_question: bool
    queries: List[str]
    reasoning: str


