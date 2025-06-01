import pydantic


SUBQUESTION_AND_ANSWER_PROMPT = """You are an expert assistant specializing in precise, well-reasoned question answering. For each task, you will receive a question and, optionally, supporting context. Your goal is to generate a subquestion that logically follows from the main question and provide a direct answer to that subquestion, accompanied by transparent, step-by-step reasoning.

Instructions:
1. **Question Analysis:** Carefully read and understand the main question. Identify key components and clarify what is being asked.
2. **Context Utilization:** If context is provided, analyze it thoroughly. Extract and summarize all relevant information that may inform your answer.
3. **Subquestion Generation:** Formulate a subquestion that logically follows from the main question. This subquestion should be specific and focused, addressing a particular aspect of the main question.
4. **Direct Answer:** Provide a concise, direct answer to the subquestion. Do not include explanations or extra commentary here.
5. **Step-by-Step Reasoning:** Document your reasoning process in clear, logical steps. Show how you move from the main question and context to the subquestion and answer, including how you resolve ambiguities or uncertainties.
6. **Output Format:** Respond in the following JSON structure with the following fields:
- "subquestion": "<The generated subquestion that logically follows from the main question>",
- "answer": "<Direct answer to the subquestion>",
- "reasoning": "<Step-by-step reasoning process to arrive at the answer>"

Here are some examples: {examples}

Now, please answer the following question:
Question: {question}
Context: {context}
"""


class SubquestionAndAnswerOutput(pydantic.BaseModel):
    subquestion: str
    answer: str
    reasoning: str

