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
Context: 
{context}
"""


class SubquestionAndAnswerOutput(pydantic.BaseModel):
    subquestion: str
    answer: str
    reasoning: str


GENERATE_SUBQUESTION_PROMPT = """You are an expert assistant specializing in precise, well-reasoned question answering. For each task, you will receive a question and, optionally, the context of reasoning steps. Your goal is to generate a subquestion that helps to clarify the main question and forward the reasoning process towards a complete answer.

Instructions:
1. **Question Analysis:** Carefully read and understand the main question. Identify key components and clarify what is being asked.
2. **Context Utilization:** If context is provided, analyze it thoroughly. Review the reasoning steps to identify gaps or areas that need further exploration.
3. **Main Question Answerability:** Consider whether context is sufficient to answer the main question. If it is, you may not need to generate a subquestion. However, if there are gaps or uncertainties, proceed to generate a subquestion.
4. **Subquestion Generation:** Formulate a subquestion that logically follows from the main question and the provided context. This subquestion should be helpful in clarifying the main question or addressing a specific aspect that requires further investigation and should follow the reasoning steps provided.
5. **Output Format:** Respond in the following JSON structure with the following fields:
- "answerable_main_question": "<A boolean indicating whether the context is sufficient to answer the main question, or 'false' if a subquestion is needed>",
- "subquestion": "<The generated subquestion that logically follows from the main question and context, if applicable>",
- "reasoning_steps": "<A brief explanation of how this subquestion helps to clarify the main question and advance the reasoning process or how the context is sufficient to answer the main question>"

Here are some examples: {examples}

Now, please determine if the main question can be answered with the provided context then generate a subquestion if necessary:
Question: {question}
Context:
{context}
"""

class SubquestionOutput(pydantic.BaseModel):
    answerable_main_question: bool
    subquestion: str
    reasoning: str
