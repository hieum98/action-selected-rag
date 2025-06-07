from typing import List, Union
from agents.retriever_agents import RetrieverAgent


class Retriever:
    def __init__(self, online_kwargs=None, offline_kwargs=None):
        self.retriever_agent = RetrieverAgent(online_kwargs=online_kwargs, offline_kwargs=offline_kwargs)

    def search(self, query: Union[str, List[str]], top_k: int = 5, return_score=False, instruction: str = ''):
        """
        Searches for the top-k relevant documents for a given query.
        Args:
            query (Union[str, List[str]]): The query or list of queries to search for.
            top_k (int): Number of top documents to retrieve.
            return_score (bool): Whether to return the scores of the retrieved documents.
            instruction (str): Additional instruction to prepend to the query.
        Returns:
            dict: A dictionary containing the retrieved documents and their scores (if requested).
        """
        return self.retriever_agent.search(query, top_k=top_k, return_score=return_score, instruction=instruction)


