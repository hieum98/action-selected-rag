import random
import copy
import json
from enum import Enum, unique
from hashlib import sha256
from typing import List, Optional, Union

import tqdm
from planners.MCTS.backbone import MCTS, Node
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
            confidence: Optional[float] = None,
            # Optional parameters
            refine_retrieved_docs: bool = True,
            max_depth: int = 5,
            golden_answer: Optional[Union[str, List[str]]] = None,
            user_question: Optional[str] = None,
            **kwargs
    ):  
        super().__init__()
        self.node_config = {
            "max_depth": max_depth,  # Maximum depth of the reasoning tree
            "refine_retrieved_docs": refine_retrieved_docs,  # Whether to refine retrieved documents before reasoning
            "golden_answer": golden_answer,  # The golden answer for the user question, if available
            "user_question": user_question,  # The main user question for USER_QUESTION nodes
        }
        self.parent = parent # Parent node in the MCTS tree, if none, this is the root node
        self.children: List["ReasoningNode"] = [] # Children nodes in the MCTS tree
        self.depth = depth
        self.generator = generator
        self.retriever = retriever
        self.node_type = node_type
        self.refine_retrieved_docs = self.node_config['refine_retrieved_docs']  # Whether to refine retrieved documents before reasoning
        self.node_content = {
            "user_question": None,  # The main user question for USER_QUESTION nodes
            "direct_answer": None,  # The direct answer for DIRECT_ANSWER nodes
            "subquestion": None,  # The subquestion of the parent node for SUBQUESTION nodes
            "subanswer": None,  # The subanswer of the parent node for SUBQUESTION nodes
            "reasoning": None,  # The reasoning content for REASONING nodes
            "confidence": None,  # The confidence score for the node, used for evaluation
        }
        if node_type == NodeType.USER_QUESTION:
            assert question is not None, "User question must be provided for USER_QUESTION nodes."
            self.node_content["user_question"] = question
            self.node_config['user_question'] = question
            self.depth = 0  # Root node has depth 0
            self.parent = None  # Root node has no parent
        elif node_type == NodeType.DIRECT_ANSWER:
            assert question is not None, "User question must be provided for DIRECT_ANSWER nodes."
            assert answer is not None, "Direct answer must be provided for DIRECT_ANSWER nodes."
            self.node_content["user_question"] = question
            self.node_content["direct_answer"] = answer
            self.node_content["reasoning"] = reasoning
            self.node_content["confidence"] = confidence
        elif node_type == NodeType.REASONING:
            assert reasoning is not None, "Reasoning content must be provided for REASONING nodes."
            self.node_content["reasoning"] = reasoning
            self.node_content["confidence"] = confidence
        elif node_type == NodeType.SUBQUESTION:
            assert question is not None, "Subquestion must be provided for SUBQUESTION nodes."
            assert answer is not None, "Subanswer must be provided for SUBQUESTION nodes."
            self.node_content["subquestion"] = question
            self.node_content['subanswer'] = answer
            self.node_content["confidence"] = confidence
        elif node_type == NodeType.RESUBQUESTION:
            assert question is not None, "Resubquestion must be provided for RESUBQUESTION nodes."
            assert answer is not None, "Resubanswer must be provided for RESUBQUESTION nodes."
            self.node_content["subquestion"] = question
            self.node_content['subanswer'] = answer
            self.node_content["confidence"] = confidence
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
            elif node.node_type == NodeType.DIRECT_ANSWER:
                reasoning_trace.append(f"{node.node_content['user_question']}\n {node.node_content['direct_answer']}")
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
        try:
            retrieval_queries = self.generator.generate_queries(question=query)
            if retrieval_queries['answerable_main_question'] or len(retrieval_queries['queries']) <= 1:
                retrieval_queries = query
            else:
                retrieval_queries = retrieval_queries['queries']
        except:
            retrieval_queries = query  # If query generation fails, use the original query
        try:
            retrieved_documents = self.retriever.search(retrieval_queries, instruction=instruction, top_k=top_k)['retrieved_docs']
            # Flatten the retrieved documents if the query is a list
            if isinstance(retrieval_queries, list):
                retrieved_documents = sum(retrieved_documents, [])
        except:
            print(f"Error retrieving documents for query: {query}")
            return ""
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
    
    def generate_direct_answer_node(self) -> List["ReasoningNode"]:
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
        for answer, reasoning, confidence in zip(output['answer'], output['reasoning'], output['confidence']):
            node = ReasoningNode(
                parent=self,
                node_type=NodeType.DIRECT_ANSWER,
                depth=self.depth + 1,
                generator=self.generator,
                retriever=self.retriever,
                question=user_question,
                confidence=confidence,
                answer=answer,
                reasoning=reasoning,
                **self.node_config  # Pass the node configuration to the new node
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
        if output['main_question_answerable']:
            # If the question is answerable, generate a direct answer node
            return self.generate_direct_answer_node()
        for next_step, confidence, need_answer in zip(output['next_step'], output['confidence'], output['need_answer']):
            if need_answer:
                supporting_information = self.get_supporting_information(query=next_step, instruction="query: ", main_query=user_question)
                reasoning_output = self.generator.generate_direct_answer(question=next_step, context=supporting_information, n=1)
                confidence = (reasoning_output['confidence'][0] + confidence) / 2.0  # Average the confidence scores
                next_step = f"{next_step}\n{reasoning_output['detailed_answer'][0]}"
            node = ReasoningNode(
                parent=self,
                node_type=NodeType.REASONING,
                depth=self.depth + 1,
                generator=self.generator,
                retriever=self.retriever,
                reasoning=next_step,
                confidence=confidence,
                **self.node_config  # Pass the node configuration to the new node
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
        path = self.get_path()
        user_question = path[0].node_content["user_question"]
        if self.node_type == NodeType.REPHASE_QUESTION:
            question = self.node_content["subquestion"] 
            supporting_information = self.get_supporting_information(query=question, instruction="query: ", main_query=user_question)
            output = self.generator.generate_direct_answer(question=question, context=supporting_information)
            highest_confidence_index = output['confidence'].index(max(output['confidence']))
            highest_confidence_node = ReasoningNode(
                parent=self,
                node_type=NodeType.SUBQUESTION,
                depth=self.depth + 1,
                generator=self.generator,
                retriever=self.retriever,
                question=question,
                answer=output['detailed_answer'][highest_confidence_index],
                confidence=output['confidence'][highest_confidence_index],
                reasoning=None,  # Subquestions do not have reasoning content
                **self.node_config  # Pass the node configuration to the new node
            )
            return [highest_confidence_node] 
        else:
            reasoning_trace = self.get_reasoning_trace()
            output = self.generator.generate_subquestion(question=user_question, context=reasoning_trace)
            main_question_answerable = output['main_question_answerable']
            if main_question_answerable:
                # If the main question is answerable, generate a direct answer node
                return self.generate_direct_answer_node()
            else:
                nodes = []
                for subquestion in output['subquestion']:
                    supporting_information = self.get_supporting_information(query=subquestion, instruction="query: ", main_query=user_question)
                    output = self.generator.generate_direct_answer(question=subquestion, context=supporting_information)
                    highest_confidence = max(output['confidence'])
                    highest_confidence_index = output['confidence'].index(highest_confidence)
                    node = ReasoningNode(
                        parent=self,
                        node_type=NodeType.SUBQUESTION,
                        depth=self.depth + 1,
                        generator=self.generator,
                        retriever=self.retriever,
                        question=subquestion,
                        answer=output['detailed_answer'][highest_confidence_index],
                        confidence=highest_confidence,
                        reasoning=None,  # Subquestions do not have reasoning content
                        **self.node_config  # Pass the node configuration to the new node
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
        nodes = []
        for answer, confidence in zip(output['reanswered_subquestion'], output['confidence']):
            node = ReasoningNode(
                parent=self,
                node_type=NodeType.RESUBQUESTION,
                depth=self.depth + 1,   
                generator=self.generator,
                retriever=self.retriever,
                question=question,
                answer=answer,
                confidence=confidence,
                reasoning=None,  # Resubquestions do not have reasoning content
                **self.node_config  # Pass the node configuration to the new node
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
        output = self.generator.rephase_question(question=question)
        nodes = []
        for rephrased_question in output['rephrased_question']:
            node = ReasoningNode(
                parent=self,
                node_type=NodeType.REPHASE_QUESTION,
                depth=self.depth + 1,
                generator=self.generator,
                retriever=self.retriever,
                question=rephrased_question,
                **self.node_config  # Pass the node configuration to the new node
            )
            nodes.append(node)
        return nodes

    def find_children(self):
        """
        Find and generate children nodes based on the current node type.
        Returns:
            List[ReasoningNode]: A list of generated child nodes based on the current node type.
        """
        if self.children:
            return self.children
        
        # Generate children based on the node type if it has not been generated yet
        if self.node_type == NodeType.USER_QUESTION:
            direct_answer_nodes = self.generate_direct_answer_node()
            reasoning_nodes = self.generate_reasoning_node()
            subquestion_nodes = self.generate_subquestion_node()
            # rephrase_question_nodes = self.generate_rephrase_question_node()
            rephrase_question_nodes = []
            children = direct_answer_nodes + reasoning_nodes + subquestion_nodes + rephrase_question_nodes
        elif self.node_type == NodeType.DIRECT_ANSWER:
            # Direct answer nodes do not generate children, they are leaf nodes
            raise ValueError("Direct answer nodes do not generate children, they are leaf nodes.")
        elif self.node_type == NodeType.REASONING:
            direct_answer_nodes = self.generate_direct_answer_node()
            subquestion_nodes = self.generate_subquestion_node()
            reasoning_nodes = self.generate_reasoning_node()
            children = direct_answer_nodes + subquestion_nodes + reasoning_nodes
        elif self.node_type == NodeType.SUBQUESTION:
            direct_answer_nodes = self.generate_direct_answer_node()
            resubquestion_nodes = self.generate_resubquestion_node()
            reasoning_nodes = self.generate_reasoning_node()
            # rephrase_question_nodes = self.generate_rephrase_question_node()
            rephrase_question_nodes = []
            subquestion_nodes = self.generate_subquestion_node()
            children = direct_answer_nodes + resubquestion_nodes + reasoning_nodes + rephrase_question_nodes + subquestion_nodes
        elif self.node_type == NodeType.RESUBQUESTION:
            direct_answer_nodes = self.generate_direct_answer_node()
            reasoning_nodes = self.generate_reasoning_node()
            subquestion_nodes = self.generate_subquestion_node()
            children = direct_answer_nodes + reasoning_nodes + subquestion_nodes
        elif self.node_type == NodeType.REPHASE_QUESTION:
            subquestion_nodes = self.generate_subquestion_node()
            children = subquestion_nodes
        else:
            raise ValueError(f"Invalid node type: {self.node_type}")
        assert len(children) > 0, f"No children generated for node type: {self.print_node()}"
        self.children = children
        return self.children
    
    def is_valid_leaf(self) -> bool:
        """
        Check if the current node is a valid leaf node.
        Returns:
            bool: True if the node is a valid leaf, False otherwise.
        """
        if self.node_type == NodeType.DIRECT_ANSWER:
            return True
        elif self.node_type == NodeType.SUBQUESTION:
            user_question = self.node_config['user_question']
            subquestion = self.node_content["subquestion"]
            is_same_question = self.generator.evaluate_same_question(question_1=user_question, question_2=subquestion)
            if is_same_question['decision']:
                # If the subquestion is the same as the user question, it is a valid leaf
                return True
        return False
    
    def is_terminal(self) -> bool:
        return self.depth > self.node_config['max_depth'] or self.is_valid_leaf()
    
    def reward(self) -> float:
        """
        Calculate the reward for the current node.
        The reward is based on the node type and content.
        Returns:
            float: The reward value for the node.
        """
        if self.is_valid_leaf():
            if self.node_type == NodeType.DIRECT_ANSWER:
                answer = self.node_content["direct_answer"]
            elif self.node_type == NodeType.SUBQUESTION:
                answer = self.node_content["subanswer"]
            reasoning = self.get_reasoning_trace()
            user_question = self.node_config['user_question']
            golden_answer = self.node_config.get('golden_answer', "Not provided")
            path = self.get_path()
            answer_confidence = self.node_content['confidence']
        else:
            # If the node is not a valid leaf, do generate direct answer and evaluate it
            direct_answer_child = self.generate_direct_answer_node()[0]
            answer = direct_answer_child.node_content["direct_answer"]
            reasoning = direct_answer_child.get_reasoning_trace()
            user_question = direct_answer_child.node_config['user_question']
            golden_answer = direct_answer_child.node_config.get('golden_answer', "Not provided")
            path = direct_answer_child.get_path()
            answer_confidence = direct_answer_child.node_content['confidence']
        
        # Answer side scoring
        if golden_answer != "Not provided":                
            output = self.generator.evaluate_answer(question=user_question, correct_answer=golden_answer, predicted_answer=answer)
            reward = output['confidence'] + output['result'] + answer_confidence
            reward = reward / 3.0 # Normalize the reward to be between 0 and 1
        else:
            query = f"{user_question}\n{answer}"
            supporting_information_for_qa = self.get_supporting_information(query=query, instruction="query: ")
            supporting_information_for_q = self.get_supporting_information(query=user_question, instruction="query: ")
            supporting_information = f"\t**Retrieved information for question and answer**\n{supporting_information_for_qa}\n\t**Retrieved information for question**\n{supporting_information_for_q}"
            output = self.generator.score_answer(question=user_question, answer=answer, context=supporting_information)
            reward = output['score'] + answer_confidence
            reward = reward / 2.0  # Normalize the reward to be between 0 and 1
        # Path confidence scoring
        path_confidence = [node.node_content['confidence'] for node in path if node.node_content['confidence'] is not None]
        if path_confidence:
            path_confidence_score = sum(path_confidence) / len(path_confidence)
        else:
            path_confidence_score = 0.0
        # LLM reasoning scoring
        reasoning_score = self.generator.score_reasoning(question=user_question, reasoning=reasoning, correct_answer=golden_answer)['score']
        reasoning_reward = (path_confidence_score + reasoning_score) / 2.0  # Normalize the reasoning reward to be between 0 and 1
        reward = (reward + reasoning_reward) / 2.0  # Combine the answer and reasoning rewards
        return reward
    
    def find_random_child(self):
        if self.is_terminal():
            return None  # If the node is terminal, return None
        node_children = self.find_children()
        random_child = random.choice(node_children) if node_children else None
        return random_child  # Return a random child node, or None if there are no children
        
    def __hash__(self):
        """
        Hash function for the ReasoningNode to use it as a key in dictionaries.
        The hash is based on the node content and type, depth, and parent node hash.
        This ensures that nodes with the same content and type will have the same hash value.
        Returns:
            int: A hash value based on the node content and type.
        """
        node_content_str = "\n".join(["{}: {}".format(k, v) for k, v in self.node_content.items()])
        node_content_str = node_content_str + f"\nnode_type: {self.node_type.value}\ndepth: {self.depth}"
        if self.parent is None:
            return  0  # If the node has no parent, return 0 as the hash value
        else:
            parent_hash = hash(self.parent) # Use the hash of the parent node to ensure uniqueness
        node_content_str = node_content_str + f"\nparent_hash: {parent_hash}"
        return int(sha256(node_content_str.encode('utf-8')).hexdigest(), 16) % (10 ** 8)  # Use a large prime number for better distribution
    
    def __eq__(self, other):
        """
        Equality check for the ReasoningNode.
        Two nodes are considered equal if they have same hash value, i.e., same content, type, depth, and parent.
        Args:
            other (ReasoningNode): The other node to compare with.
        Returns:
            bool: True if the nodes are equal, False otherwise.
        """
        if not isinstance(other, ReasoningNode):
            return False
        same_hash = hash(self) == hash(other)
        same_content = self.node_content == other.node_content
        same_type = self.node_type == other.node_type
        same_depth = self.depth == other.depth
        return same_hash and same_content and same_type and same_depth
        
    def print_node(self):
        """
        Print the node content in a readable format.
        """
        node_content = copy.deepcopy(self.node_content)
        node_content['node_type'] = self.node_type.value
        node_content['depth'] = self.depth
        node_content['parent'] = hash(self.parent) if self.parent else None
        print("Node content:")
        for key, value in node_content.items():
            print(f"{key}: {value}")
                

if __name__=='__main__':
    from planners.MCTS.reasoning_node import *

    # Example usage
    generator_online_model_kwargs = {
        'model_name': 'qwen3-32b',
        'url': 'http://n0999.talapas.uoregon.edu:30000/v1',
        'api_key': 'None',
        'concurrency': 64,
    }
    generate_kwargs = {
        'temperature': 0.6,
        'n': 1, # should be odd number ás it is used for majority voting
        'top_p': 0.95,
        'max_tokens': 8192,
        'top_k': 20,
        'repetition_penalty': 1.1,
        'logprobs': 1,
        'tensor_parallel_size': 1,
    }
    generator = Generator(online_model_kwargs=generator_online_model_kwargs, generate_kwargs=generate_kwargs, verbose=True)

    retriever_online_kwargs = {
        "url": "http://n0999.talapas.uoregon.edu:5000/search",
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
        refine_retrieved_docs=True,
    )

    print("Root node created:")
    root_node.print_node()
    # print("Finding children nodes...")
    # children = root_node.find_children()
    # print(f"Found {len(children)} children nodes:")
    # for child in children:
    #     child.print_node()
    # print("Finding random child node...")
    # random_child = root_node.find_random_child()
    # if random_child:
    #     print("Random child node found:")
    #     random_child.print_node()
    # else:
    #     print("No random child node found.")
    # print("Generating direct answer node...")
    # direct_answer_nodes = root_node.generate_direct_answer_node()
    # print(f"Generated {len(direct_answer_nodes)} direct answer nodes:")
    # for node in direct_answer_nodes:
    #     node.print_node()
    print("Generating reasoning node...")
    reasoning_nodes = root_node.generate_reasoning_node()
    print(f"Generated {len(reasoning_nodes)} reasoning nodes:")
    for node in reasoning_nodes:
        node.print_node()
    # print("Generating subquestion node...")
    # subquestion_nodes = root_node.generate_subquestion_node()
    # print(f"Generated {len(subquestion_nodes)} subquestion nodes:")
    # for node in subquestion_nodes:
    #     node.print_node()
    # print("Generating resubquestion node...")
    # resubquestion_nodes = subquestion_nodes[0].generate_resubquestion_node()  # Generate resubquestion from the first subquestion node
    # print(f"Generated {len(resubquestion_nodes)} resubquestion nodes:")
    # for node in resubquestion_nodes:
    #     node.print_node()
    # print("Generating rephrase question node...")
    # rephrase_question_nodes = root_node.generate_rephrase_question_node()
    # print(f"Generated {len(rephrase_question_nodes)} rephrase question nodes:")
    # for node in rephrase_question_nodes:
    #     node.print_node()
    # print("Finding children nodes again...")
    # children = root_node.find_children()
    # print(f"Found {len(children)} children nodes after generating all types:")
    # for child in children:
    #     child.print_node()
    # print("Checking if the root node is terminal...")
    # is_terminal = root_node.is_terminal()
    # print(f"Is the root node terminal? {is_terminal}")
    # print("Checking if the root node is a valid leaf...")
    # is_valid_leaf = root_node.is_valid_leaf()
    # print(f"Is the root node a valid leaf? {is_valid_leaf}")
    # print("Calculating reward for the root node...")
    # reward = root_node.reward()
    # print(f"Reward for the root node: {reward}")        



