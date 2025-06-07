from enum import Enum, unique
from typing import List, Optional, Union
from planners.MCTS.backbone import Node
from planners.generator import Generator
from planners.retriever import Retriever


@unique
class NodeType(Enum):
    USER_QUESTION = "USER_QUESTION"  # Node type for user question, i.e., the root node
    DIRECT_ANSWER = "DIRECT_ANSWER"  # Node type for direct answer, i.e., the leaf node
    REASONING = "REASONING"  # Node type for reasoning, i.e., the intermediate node
    SUBQUESTION = "SUBQUESTION"  # Node type for subquestion and answers i.e., the intermediate node
    RESUBQUESTION = "RESUBQUESTION" # Node type for reanswering subquestions, i.e., the intermediate node
    REPHASE_QUESTION = "REPHRASE"  # Node type for rephrased question, i.e., the intermediate node

class ReasoningNode(Node):
    """
    A node in the MCTS tree that represents a reasoning step.
    Args:
        parent (ReasoningNode): The parent node in the MCTS tree.
        node_type (NodeType): The type of the node, e.g., USER_QUESTION, DIRECT_ANSWER, etc.
        depth (int): The depth of the node in the MCTS tree.
        generator (Generator): The generator component to generate follow-up reasoning steps.
        retriever (Retriever): The retriever component to retrieve relevant information for reasoning.
        question (Optional[str]): The question associated with the node, required for USER_QUESTION and SUBQUESTION nodes.
        answer (Optional[str]): The answer associated with the node, required for DIRECT_ANSWER and SUBQUESTION nodes.
        reasoning (Optional[str]): The reasoning content associated with the node, required for REASONING nodes.
    """
    def __init__(
            self,
            # Node parameters
            parent: "ReasoningNode",
            node_type: NodeType,
            depth: int,
            # Components
            generator: Generator,
            retriever: Retriever,
            question: Optional[str] = None,
            answer: Optional[str] = None,
            reasoning: Optional[str] = None,
            # Optional parameters
            refine_retrieved_docs: bool = True,
    ):  
        super().__init__()
        self.parent = parent # Parent node in the MCTS tree, if none, this is the root node
        self.children: List["ReasoningNode"] = [] # Children nodes in the MCTS tree
        self.depth = depth
        self.generator = generator
        self.retriever = retriever
        self.node_type = node_type
        self.refine_retrieved_docs = refine_retrieved_docs  # Whether to refine retrieved documents before reasoning
        self.node_content = {
            "user_question": None,  # The main user question for USER_QUESTION nodes
            "direct_answer": None,  # The direct answer for DIRECT_ANSWER nodes
            "subquestion": None,  # The subquestion of the parent node for SUBQUESTION nodes
            "subanswer": None,  # The subanswer of the parent node for SUBQUESTION nodes
            "reasoning": None,  # The reasoning content for REASONING nodes
        }
        if node_type == NodeType.USER_QUESTION:
            assert question is not None, "User question must be provided for USER_QUESTION nodes."
            self.node_content["user_question"] = question
            self.depth = 0  # Root node has depth 0
            self.parent = None  # Root node has no parent
        elif node_type == NodeType.DIRECT_ANSWER:
            assert question is not None, "User question must be provided for DIRECT_ANSWER nodes."
            assert answer is not None, "Direct answer must be provided for DIRECT_ANSWER nodes."
            self.node_content["user_question"] = question
            self.node_content["direct_answer"] = answer
            self.node_content["reasoning"] = reasoning
        elif node_type == NodeType.REASONING:
            assert reasoning is not None, "Reasoning content must be provided for REASONING nodes."
            self.node_content["reasoning"] = reasoning
        elif node_type == NodeType.SUBQUESTION:
            assert question is not None, "Subquestion must be provided for SUBQUESTION nodes."
            assert answer is not None, "Subanswer must be provided for SUBQUESTION nodes."
            self.node_content["subquestion"] = question
            self.node_content['subanswer'] = answer
        elif node_type == NodeType.RESUBQUESTION:
            assert question is not None, "Resubquestion must be provided for RESUBQUESTION nodes."
            assert answer is not None, "Resubanswer must be provided for RESUBQUESTION nodes."
            self.node_content["subquestion"] = question
            self.node_content['subanswer'] = answer
        elif node_type == NodeType.REPHASE_QUESTION:
            assert question is not None, "Rephrased question must be provided for REPHASE_QUESTION nodes."
            self.node_content["subquestion"] = question
        else:
            raise ValueError(f"Invalid node type: {node_type}")
    
    def get_path(self) -> List["ReasoningNode"]:
        """
        Get the path from the root node to the current node.
        Returns:
            List[ReasoningNode]: A list of nodes from the root to the current node.
        """
        path = []
        current_node = self
        while current_node is not None:
            path.append(current_node)
            current_node = current_node.parent
        return path[::-1] # Reverse the path to get it from root to current node
    
    def get_reasoning_trace(self) -> str:
        """
        Get the reasoning trace from the root node to the current node.
        Returns:
            str: A string representation of the reasoning trace, including all reasoning steps and subquestions.
        """
        path = self.get_path()
        reasoning_trace = []
        for i, node in enumerate(path):
            if node.node_type == NodeType.REASONING:
                reasoning_trace.append(node.node_content["reasoning"])
            elif node.node_type == NodeType.SUBQUESTION:
                step_content = f"{node.node_content['subquestion']}\n {node.node_content['subanswer']}"
                reasoning_trace.append(step_content)
            elif node.node_type == NodeType.RESUBQUESTION:
                step_content = f"{node.node_content['subquestion']}\n {node.node_content['subanswer']}"
                # If the previous node is a SUBQUESTION, replace the subanswer with the resubanswer
                reasoning_trace[-1] = step_content
        trace = ""
        for i, step in enumerate(reasoning_trace):
            trace += f"Step {i+1}: {step}\n"
        return trace
    
    def get_supporting_information(self, query: str, instruction: str = 'query: ', top_k: int = None, main_query: str = ""):
        """
        Retrieve supporting information for the current reasoning step.
        Args:
            query (str): The query to retrieve supporting information for.
            instruction (str): The instruction to prepend to the query, default is 'query: '.
            top_k (int): The number of top documents to retrieve, default is None (use retriever's default).
            main_query (str): The main user question for context, default is an empty string.
        Returns:
            str: A string representation of the retrieved supporting information.
        """
        retrieved_documents = self.retriever.search(query, instruction=instruction, top_k=top_k)['retrieved_docs']
        if self.refine_retrieved_docs:
            output = self.generator.extract_information_from_retrieved_docs(
                question=main_query,
                document=retrieved_documents,
                current_step_objective=query,
            )
            information = []
            for relevant, info in zip(output['decision'], output['extracted_information']):
                if relevant:
                    information.append(info)
            information = ["Retrieved information {}: {}".format(i+1, info) for i, info in enumerate(information)]
        else:
            information = ["Retrieved document {}: {}".format(i+1, doc) for i, doc in enumerate(retrieved_documents)]
        information = "\n".join(information)
        return information
    
    def generate_direct_answer_node(self):
        """
        Generate a direct answer node from the current node.
        Returns:
            List[ReasoningNode]: A new node of type DIRECT_ANSWER with the generated answer.
        """
        assert self.node_type not in [NodeType.DIRECT_ANSWER, NodeType.REPHASE_QUESTION], "Direct answer nodes cannot be generated from DIRECT_ANSWER or REPHASE_QUESTION nodes."
        # Get the path from the root to the current node
        path = self.get_path()
        user_question = path[0].node_content["user_question"]
        reasoning_trace = self.get_reasoning_trace()
        supporting_information = self.get_supporting_information(
            query=user_question,
            instruction="query: ",
            main_query=user_question,
            )
        context = f"\t**Reasoning trace** \n{reasoning_trace}\n\t**Supporting information** \n{supporting_information}"
        output = self.generator.generate_direct_answer(question=user_question, context=context)
        nodes = []
        for answer, reasoning in zip(output['answer'], output['reasoning']):
            node = ReasoningNode(
                parent=self,
                node_type=NodeType.DIRECT_ANSWER,
                depth=self.depth + 1,
                generator=self.generator,
                retriever=self.retriever,
                question=user_question,
                answer=answer,
                reasoning=reasoning,
            )
            nodes.append(node)
        return nodes
    
    def generate_reasoning_node(self):
        """
        Generate a reasoning node from the current node.
        Returns:
            List[ReasoningNode]: A new node of type REASONING with the generated reasoning content.
        """
        assert self.node_type not in [NodeType.DIRECT_ANSWER, NodeType.REPHASE_QUESTION], "Reasoning nodes cannot be generated from DIRECT_ANSWER or REPHASE_QUESTION nodes."
        path = self.get_path()
        user_question = path[0].node_content["user_question"]
        reasoning_trace = self.get_reasoning_trace()
        output = self.generator.generate_follow_up_reasoning(question=user_question, context=reasoning_trace)
        nodes = []
        for next_step in output['next_step']:
            node = ReasoningNode(
                parent=self,
                node_type=NodeType.REASONING,
                depth=self.depth + 1,
                generator=self.generator,
                retriever=self.retriever,
                reasoning=next_step,
            )
            nodes.append(node)
        return nodes
    
    def generate_subquestion_node(self):
        """
        Generate a subquestion node from the current node.
        Returns:
            List[ReasoningNode]: A new node of type SUBQUESTION with the generated subquestion and answer.
        """
        assert self.node_type != NodeType.DIRECT_ANSWER, "Subquestion nodes cannot be generated from DIRECT_ANSWER nodes."
        if self.node_type == NodeType.REPHASE_QUESTION:
            question = self.node_content["subquestion"] 
            supporting_information = self.get_supporting_information(query=question, instruction="query: ")
            output = self.generator.generate_direct_answer(question=question, context=supporting_information)
            answers = output['answer']
            nodes = []
            for answer in answers:
                node = ReasoningNode(
                    parent=self,
                    node_type=NodeType.SUBQUESTION,
                    depth=self.depth + 1,
                    generator=self.generator,
                    retriever=self.retriever,
                    question=question,
                    answer=answer,
                    reasoning=None,  # Subquestions do not have reasoning content
                )
                nodes.append(node)
            return nodes
        else:
            path = self.get_path()
            user_question = path[0].node_content["user_question"]
            reasoning_trace = self.get_reasoning_trace()
            output = self.generator.generate_subquestion(question=user_question, context=reasoning_trace)
            main_question_answerable = output['main_question_answerable']
            if main_question_answerable:
                # If the main question is answerable, generate a direct answer node
                return self.generate_direct_answer_node()
            else:
                for subquestion in output['subquestion']:
                    supporting_information = self.get_supporting_information(query=subquestion, instruction="query: ")
                    output = self.generator.generate_direct_answer(question=subquestion, context=supporting_information)
                    nodes = []
                    for answer in output['answer']:
                        node = ReasoningNode(
                            parent=self,
                            node_type=NodeType.SUBQUESTION,
                            depth=self.depth + 1,
                            generator=self.generator,
                            retriever=self.retriever,
                            question=subquestion,
                            answer=answer,
                            reasoning=None,  # Subquestions do not have reasoning content
                        )
                        nodes.append(node)
                return nodes

    def generate_resubquestion_node(self):
        """
        Generate a resubquestion node from the current node.
        Returns:
            List[ReasoningNode]: A new node of type RESUBQUESTION with the generated resubquestion and answer.
        """
        assert self.node_type == NodeType.SUBQUESTION, "Resubquestion nodes can only be generated from SUBQUESTION nodes."
        question = self.node_content["subquestion"]
        answer = self.node_content["subanswer"]
        retriever_query = f"{question}\n{answer}"
        
        supporting_information = self.get_supporting_information(query=retriever_query, instruction="query: ")
        output = self.generator.reanswer_subquestion(question=question, answer=answer, context=supporting_information)
        answers = output['reanswered_subquestion']       
        nodes = []
        for answer in answers:
            node = ReasoningNode(
                parent=self,
                node_type=NodeType.RESUBQUESTION,
                depth=self.depth + 1,   
                generator=self.generator,
                retriever=self.retriever,
                question=question,
                answer=answer,
                reasoning=None,  # Resubquestions do not have reasoning content
            )
            nodes.append(node)
        return nodes
    
    def generate_rephrase_question_node(self):
        """
        Generate a rephrase question node from the current node.
        Returns:
            List[ReasoningNode]: A new node of type REPHASE_QUESTION with the rephrased question.
        """
        assert self.node_type not in [NodeType.DIRECT_ANSWER, NodeType.REASONING], "Rephrase question nodes cannot be generated from DIRECT_ANSWER or REASONING nodes."
        reasoning_trace = self.get_reasoning_trace()
        if self.node_type == NodeType.USER_QUESTION:
            question = self.node_content["user_question"]
        else:
            question = self.node_content["subquestion"]
        output = self.generator.rephase_question(question=question, context=reasoning_trace)
        nodes = []
        for rephrased_question in output['rephrased_question']:
            node = ReasoningNode(
                parent=self,
                node_type=NodeType.REPHASE_QUESTION,
                depth=self.depth + 1,
                generator=self.generator,
                retriever=self.retriever,
                question=rephrased_question,
            )
            nodes.append(node)
        return nodes

    def find_children(self):
        if self.node_type == NodeType.USER_QUESTION:
            pass


if __name__=='__main__':
    # Example usage
    generator_online_model_kwargs = {
        'model_name': 'qwen3-32b',
        'url': 'http://n0998.talapas.uoregon.edu:30000/v1',
        'api_key': 'None',
        'concurrency': 64,
    }
    generate_kwargs = {
        'temperature': 0.6,
        'n': 3, # should be odd number ás it is used for majority voting
        'top_p': 0.95,
        'max_tokens': 8192,
        'top_k': 20,
        'repetition_penalty': 1.1,
        'logprobs': 1,
        'tensor_parallel_size': 1,
    }
    generator = Generator(online_model_kwargs=generator_online_model_kwargs, generate_kwargs=generate_kwargs, verbose=True)

    retriever_online_kwargs = {
        "url": "http://n0998.talapas.uoregon.edu:5000/search",
        "retrieval_topk": 5,
        "query_instruction": "query: ",
    }
    retriever = Retriever(online_kwargs=retriever_online_kwargs)

    question = "Are director of film Move (1970 Film) and director of film Méditerranée (1963 Film) from the same country?"
    root_node = ReasoningNode(
        parent=None,
        node_type=NodeType.USER_QUESTION,
        depth=0,
        generator=generator,
        retriever=retriever,
        question=question,
    )



