import pydantic


ONE_NEXT_STEP_PROMPT = """You are an expert assistant specializing in precise, well-reasoned question answering. For each task, you will receive a question and, optionally, current context. Your goal is to deliver a single next reasoning step that logically follows from the context to advance towards a complete answer to the question.

Instructions:
1. **Question Analysis:** Carefully read and understand the question. Identify key components and clarify what is being asked.
2. **Context Analysis:**  If context are provided, analyze it thoroughly. Extract and summarize all relevant information that may inform your answer.
3. **Information Gap Identification:** Identify any gaps in the context that need to be addressed to move closer to a complete answer. Formulate a specific next step that would help fill these gaps.
4. **Output Format:** Respond in the following JSON structure with following fields:
- "next_step": "<A single next reasoning step that logically follows from the context>",
- "justification": "<A brief explanation of why this step is necessary and how it contributes to answering the question>"

Here are some examples: {examples}

Now, please generate a single next reasoning step to advance towards answering the question:
Question: {question}
Context: 
{context}
"""

class OneNextStepOutput(pydantic.BaseModel):
    next_step: str
    justification: str

