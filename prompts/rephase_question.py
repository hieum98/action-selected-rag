import pydantic


REPHRASE_QUESTION_PROMPT = """You are an expert assistant specializing in precise, well-reasoned question rephrasing. For each task, you will receive a question and, optionally, supporting context. Your goal is to rephrase the question to make it clearer, more specific, or more focused while retaining its original intent.

Instructions:
1. **Question Analysis:** Carefully read and understand the question. Identify key components and clarify what is being asked.
2. **Context Utilization:** If context is provided, analyze it thoroughly. Extract and summarize all relevant information that may inform your rephrasing.
3. **Rephrasing:** Reformulate the question to enhance clarity, specificity, or focus. Ensure that the rephrased question retains the original intent and is suitable for further analysis or answering.
4. **Output Format:** Respond in the following JSON structure with the following fields:
- "rephrased_question": "<The rephrased question that retains the original intent>",
- "reasoning": "<Step-by-step reasoning process explaining how you arrived at the rephrased question>"

Here are some examples: {examples}

Now, please rephrase the following question:
Question: {question}
Context: {context}
"""

class RephraseQuestionOutput(pydantic.BaseModel):
    rephrased_question: str
    reasoning: str

