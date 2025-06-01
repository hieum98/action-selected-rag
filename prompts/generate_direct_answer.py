import pydantic


DIRECT_ANSWER_PROMPT = """You are an expert assistant specializing in precise, well-reasoned question answering. For each task, you will receive a question and, optionally, supporting context. Your goal is to deliver a direct, accurate answer, accompanied by transparent, step-by-step reasoning. 

Instructions:
1. **Question Analysis:** Carefully read and understand the question. Identify key components and clarify what is being asked.
2. **Context Utilization:**  If context is provided, analyze it thoroughly. Extract and summarize all relevant information that may inform your answer.
3. **Information Gap Identification:** If the context does not fully answer the question, identify missing information. Formulate specific follow-up queries that would help fill these gaps. Attempt to answer these queries based on your own knowledge.
4. **Step-by-Step Reasoning:** Document your reasoning process in clear, logical steps. Show how you move from question and context to answer, including how you resolve ambiguities or uncertainties.
5. **Direct Answer:** Provide a concise, direct answer to the question. Do not include explanations or extra commentary here.
6. **Output Format:** Respond in the following JSON structure with following fields:
- "additional_information": "<Any extra insights, caveats, or related facts (optional)>"
- "reasoning": "<Step-by-step reasoning process to arrive at the answer>",
- "answer": "<Direct answer to the question>",

Here are some examples: {examples}

Now, please answer the following question:
Question: {question}
Context: {context}
"""


class DirectAnswerOutput(pydantic.BaseModel):
    additional_information: str
    reasoning: str
    answer: str

