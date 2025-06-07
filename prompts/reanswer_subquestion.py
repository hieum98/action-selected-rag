import pydantic


REANSWER_PROMPT = """You are an expert assistant specializing in precise, well-reasoned question answering. For each task, you will receive a question, an answer for this question and, optionally, supporting context. Your goal is to reanswer the question based on the provided answer and context, ensuring that your response is accurate, concise, and well-reasoned. Please note that the answer may not be correct, and you should verify it against the context provided.

Instructions:
1. **Question Analysis:** Carefully read and understand the main question. Identify key components and clarify what is being asked.
2. **Context Utilization:** If context is provided, analyze it thoroughly. Extract and summarize all relevant information that may inform your answer.
3. **Answer Verification:** Review the provided answer critically. Determine if it is correct based on the context and your own knowledge. If the answer is incorrect or incomplete, identify the specific issues.
4. **Reanswering:** If the provided answer is correct, simply restate it. If it is incorrect or incomplete, provide a corrected answer based on the context and your reasoning.
5. **Step-by-Step Reasoning:** Document your reasoning process in clear, logical steps. Show how you arrive at the reanswer, including how you resolve ambiguities or uncertainties.
6. **Output Format:** Respond in the following JSON structure with the following fields:
- "reanswer": "<The reanswered question based on the provided answer and context>",
- "reasoning": "<Step-by-step reasoning process to arrive at the reanswer>",

Here are some examples: {examples}

Now, please reanswer the following question:
Question: {question}
Answer: {answer}
Context: 
{context}
"""


class ReanswerOutput(pydantic.BaseModel):
    reanswer: str
    reasoning: str


