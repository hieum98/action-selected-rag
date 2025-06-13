import pydantic


ONE_NEXT_STEP_PROMPT = """You are an expert assistant specializing in multi-hop reasoning and sequential question answering. Your task is to analyze the current reasoning state and generate the optimal next reasoning step that advances toward a complete answer, or determine if sufficient information exists to conclude the reasoning process.

## Core Objectives:
1. **Assess Context Sufficiency:** Determine if the provided context contains sufficient information to answer the main question completely and accurately.
2. **Strategic Next Step Generation:** When reasoning should continue, identify the single most productive next reasoning step that bridges critical knowledge gaps.

## Decision Framework:
**Context Sufficiency Criteria:**
- All question entities and relationships are sufficiently addressed
- Logical reasoning chain connects context to a definitive answer
- No critical inferential gaps remain that would compromise answer confidence
- Temporal, causal, or comparative requirements are fully satisfied
**Next Step Quality Criteria:**
- **Logical Progression:** Follows naturally from current reasoning state
- **Gap Targeting:** Addresses the most critical missing information
- **Specificity:** Focuses on a single, well-defined reasoning operation  
- **Productivity:** Maximizes progress toward final answer resolution
- **Actionability:** Can be executed with available information or reasonable inference

## Instructions:
1. **Question Decomposition:** Parse the question type (factual, comparative, causal, temporal, multi-hop) and identify required reasoning operations.
2. **Context State Analysis:** 
    - Extract all established facts and relationships
    - Map current knowledge against question requirements
    - Identify the furthest point reached in the reasoning chain
3. **Gap Assessment:** Determine the most critical missing element:
    - **Entity gaps:** Missing key entities or their properties
    - **Relationship gaps:** Undefined connections between known entities
    - **Logical gaps:** Missing inferential steps in the reasoning chain
    - **Evidence gaps:** Insufficient support for required conclusions
4. Generate a step using these patterns:
    - **For entity resolution:** "Determine [specific property/attribute] of [entity]"
    - **For relationship establishment:** "Establish how [entity A] relates to [entity B] in [context]"  
    - **For logical inference:** "Apply [reasoning type] to conclude [specific outcome]"
    - **For evidence gathering:** "Verify [claim] using [available evidence]"
5. **Validation:** Ensure the next step is the minimal sufficient action to advance reasoning optimally.

## Output Format:
Respond in the following JSON structure with the following fields:
- "answerable_main_question": <boolean value indicating whether the main question can be answered with the provided context>,
- "next_step": "<The generated next reasoning step that logically follows from the current context>",
- "justification": "<Reasoning process explaining how you arrived at the decision and next step>"
- "confidence": "<Confidence level of the next step, one of 'high', 'medium', or 'low'>"
- "inference_type": "<Type of reasoning gap identified, one of 'entity', 'relationship', 'logical', 'evidence', or 'null'>"

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
    inference_type: str = pydantic.Field(
        ...,
        pattern=r"^(entity|relationship|logical|evidence|null)$",
    )
        

