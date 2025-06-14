from typing import Optional, List, Union
import datasets

from planners.generator import Generator
from planners.retriever import Retriever


def answer_question(question: str, goden_answer: Union[str, List[str]], max_rollout: int = 20):
    n_rollout = 0
    while n_rollout < max_rollout:
        # Generate query for the retriever
        queries = generator.generate_queries(question=question)['queries']
        if not queries:
            queries = [question]
        # Retrieve relevant documents for each query
        retrieved_documents = retriever.search(queries, instruction='query: ', top_k=5)['retrieved_docs']
        if isinstance(queries, list):
            retrieved_documents = sum(retrieved_documents, [])
        retrieved_info = [f"Retrieved document {i+1}: {doc}" for i, doc in enumerate(retrieved_documents)]
        retrieved_info = "\n".join(retrieved_info)
        # Generate a direct answer from the retrieved documents
        answer = generator.generate_direct_answer(question=question, context=retrieved_info, n=1)['answer'][0]
        # Check if the answer matches the golden answer
        is_correct = generator.evaluate_answer(question=question, correct_answer=goden_answer, predicted_answer=answer)['result']
        if is_correct:
            return n_rollout
        else:
            n_rollout = n_rollout + 1
    return max_rollout
    

if __name__ == "__main__":
    # Example usage
    generator_online_model_kwargs = {
        'model_name': 'qwen3-8b',
        'url': 'http://n0999.talapas.uoregon.edu:30000/v1',
        'api_key': 'None',
        'concurrency': 64,
    }
    generate_kwargs = {
        'temperature': 0.95, # Higher temperature for more diverse answers
        'n': 1, 
        'top_p': 0.95,
        'max_tokens': 8192,
        'top_k': 20,
        'repetition_penalty': 1.1,
        'logprobs': 1,
        'tensor_parallel_size': 1,
    }
    generator = Generator(online_model_kwargs=generator_online_model_kwargs, generate_kwargs=generate_kwargs, verbose=False, use_cache=False)

    retriever_online_kwargs = {
        "url": "http://n0998.talapas.uoregon.edu:5000/search",
        "retrieval_topk": 5,
        "query_instruction": "query: ",
    }
    retriever = Retriever(online_kwargs=retriever_online_kwargs, verbose=False)

    data = datasets.load_dataset("RUC-NLPIR/FlashRAG_datasets", name="2wikimultihopqa")
    # Concatenate all splits into one dataset
    split_names = data.keys()
    all_data = []
    for split in split_names:
        split = data[split].map(lambda x: {'split': split})
        all_data.append(split)
    all_data = datasets.concatenate_datasets(all_data)
    # Try to answer each question in the dataset to sellect hard data
    all_data = all_data.map(
        lambda x: {'n_rollout': answer_question(question=x['question'], goden_answer=x['golden_answers'], max_rollout=20)},
        num_proc=16
    )

    # Save the data
    all_data.save_to_disk("data/2wikimultihopqa")