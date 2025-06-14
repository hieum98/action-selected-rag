import copy
import os
from typing import Any, Dict, List
import json
from hashlib import sha256
from functools import partial
from pydantic import BaseModel
import openai
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from pebble import ThreadPool
import tqdm
import outlines
from outlines.samplers import multinomial
from transformers import AutoModelForCausalLM, AutoTokenizer


class OpenAIClient:
    def __init__(
            self, 
            model_name: str,
            url: str, 
            api_key: str = 'None',
            concurrency: int = 64,
            **generate_kwargs: Dict[str, Any]
            ):
        self.client = openai.Client(base_url=url, api_key=api_key)
        print(f"Using OpenAI API at {url} with model {model_name}")
        self.model_name = model_name
        # Generation parameters
        self.temperature = generate_kwargs.get('temperature', 0.7)
        self.num_samples = generate_kwargs.get('n', 1)
        self.top_p = generate_kwargs.get('top_p', 0.8)
        self.max_tokens = generate_kwargs.get('max_tokens', 8192) # default max tokens to generate
        self.presence_penalty = generate_kwargs.get('repetition_penalty', 1.5)
        self.top_k = generate_kwargs.get('top_k', 20)
       
        self.concurrency = concurrency
    
    def generate(self, input: Dict[str, Any], **kwargs) -> List[Dict[str, str]]:
        """
        Generate a response from the OpenAI API.
        """
        messages = input['messages']
        index = input['index']
        # Here json_schema should be a dictionary in the format:
        response_class = input.get('json_schema', None)
        if response_class is not None:
            assert issubclass(response_class, BaseModel), "response_class must be a subclass of BaseModel"
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_class.__name__,
                    "schema": response_class.model_json_schema()
                    }
            }
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                top_p=kwargs.get('top_p', self.top_p),
                n=kwargs.get('n', self.num_samples),
                presence_penalty=kwargs.get('repetition_penalty', self.presence_penalty),
                response_format=response_format,
                extra_body={
                    "top_k": kwargs.get('top_k', self.top_k),
                    "chat_template_kwargs": {"enable_thinking": True},
                }
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                top_p=kwargs.get('top_p', self.top_p),
                n=kwargs.get('n', self.num_samples),
                extra_body={
                    "top_k": kwargs.get('top_k', self.top_k),
                    "chat_template_kwargs": {"enable_thinking": True},
                }
            )
    
        # Extract the generated text from the response
        responses = []
        for choice in response.choices:
            try:
                cot_reasoning = choice.message.reasoning_content
            except AttributeError:
                cot_reasoning = None
            if response_class is not None:
                try:
                    response_object = response_class.model_validate_json(choice.message.content)
                except Exception as e:
                    print(f"Error validating response: {e}")
                    print(f"Response: {choice.message.content}")
                    response_object = None
            else:
                response_object = choice.message.content
            responses.append({
                'cot_reasoning': cot_reasoning,
                'output': response_object,
            })
        return {
            'index': index,
            'input': messages,
            'output': responses,
        }

    def batch_generate(
            self, 
            batch: List[List[Dict[str, str]]],
            response_object: BaseModel = None,
            **kwargs
            ) -> List[List[Dict[str, str]]]:
        """
        Generate a batch of responses from the OpenAI API.
        """
        if response_object is not None:
            assert issubclass(response_object, BaseModel), "response_object must be a subclass of BaseModel"
        batch = [{'index': i, 'messages': messages, 'json_schema': response_object} for i, messages in enumerate(batch)]
        max_workers = min(self.concurrency, len(batch))
        generate_fn = partial(self.generate, **kwargs)
        with ThreadPool(max_workers=max_workers) as pool:
            future = pool.map(generate_fn, batch)
            outputs = list(tqdm.tqdm(future.result(), total=len(batch), desc=f"Generating responses from {self.model_name} with OpenAI API"))
        assert len(outputs) == len(batch), f"Expected {len(batch)} outputs, but got {len(outputs)}"
        # Convert outputs to dict with index as key
        outputs = {output['index']: output for output in outputs}
        responses = []
        for item in batch:
            index = item['index']
            responses.append(outputs[index])
        return responses
    

class HFAgent:
    def __init__(
            self, 
            model_name: str,
            **generate_kwargs: Dict[str, Any]
            ):
        self.model = outlines.models.transformers(model_name, device="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Generation parameters
        self.temperature = generate_kwargs.get('temperature', 0.8)
        self.num_samples = generate_kwargs.get('n', 1)
        self.top_p = generate_kwargs.get('top_p', 0.95)
        self.top_k = generate_kwargs.get('top_k', 50)
        self.max_tokens = generate_kwargs.get('max_tokens', 1024) # default max tokens to generate
        self.sampler = multinomial(samples=self.num_samples, temperature=self.temperature, top_p=self.top_p, top_k=self.top_k)
    
    def generate(self, input: Dict[str, Any], **kwargs) -> List[Dict[str, str]]:
        """
        Generate a response from the Hugging Face model.
        """
        response_object = input.get('json_schema', None)
        if response_object is not None:
            assert issubclass(response_object, BaseModel), "response_object must be a subclass of BaseModel"
            generator = outlines.generate.json(self.model, response_object, sampler=self.sampler)
        else:
            generator = outlines.generate.text(self.model, sampler=self.sampler)
        messages = input['messages']
        input = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = generator(input, max_tokens=self.max_tokens,)
        return {
            'input': messages,
            'output': outputs,
        }
    
    def batch_generate(
            self, 
            batch: List[List[Dict[str, str]]],
            response_object: BaseModel = None,
            **kwargs
            ) -> List[List[Dict[str, str]]]:
        """
        Generate a batch of responses from the Hugging Face model.
        """
        input = self.tokenizer.apply_chat_template(batch, tokenize=False, add_generation_prompt=True)
        if response_object is not None:
            assert issubclass(response_object, BaseModel), "response_object must be a subclass of BaseModel"
            generator = outlines.generate.json(self.model, response_object, sampler=self.sampler)
        else:
            generator = outlines.generate.text(self.model, sampler=self.sampler)
        outputs = generator(input, max_tokens=self.max_tokens)
        assert len(outputs) == len(batch), f"Expected {len(batch)} outputs, but got {len(outputs)}"
        responses = []
        for messages, output in zip(batch, outputs):
            responses.append({
                'input': messages,
                'output': output,
            })
        return responses


class vLLMAgent:
    def __init__(
            self, 
            model_name: str,
            **generate_kwargs: Dict[str, Any]
            ):
        self.model = LLM(
            model=model_name,
            tensor_parallel_size=generate_kwargs.get('tensor_parallel_size', 1),
            trust_remote_code=True,
            max_num_seqs=256,
            swap_space=16,
            max_model_len=16384,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Generation parameters
        self.temperature = generate_kwargs.get('temperature', 0.8)
        self.num_samples = generate_kwargs.get('n', 1)
        self.top_p = generate_kwargs.get('top_p', 0.95)
        self.top_k = generate_kwargs.get('top_k', 50)
        self.repetition_penalty = generate_kwargs.get('repetition_penalty', 1.1)
        self.max_tokens = generate_kwargs.get('max_tokens', 1024)
        self.logprobs = generate_kwargs.get('logprobs', 1)
    
    def generate(self, input: Dict[str, Any], **kwargs) -> List[Dict[str, str]]:
        """
        Generate a response from the vLLM model.
        """
        response_object = input.get('json_schema', None)
        if response_object is not None:
            assert issubclass(response_object, BaseModel), "response_object must be a subclass of BaseModel"
            json_schema = response_object.model_json_schema()
            guided_decoding_params = GuidedDecodingParams(json=json_schema)
        else:
            guided_decoding_params = None
        sampling_params = SamplingParams(
            temperature=kwargs.get('temperature', self.temperature),
            guided_decoding=guided_decoding_params,
            top_p=kwargs.get('top_p', self.top_p),
            top_k=kwargs.get('top_k', self.top_k),
            repetition_penalty=kwargs.get('repetition_penalty', self.repetition_penalty),
            n= kwargs.get('n', self.num_samples),
            logprobs=kwargs.get('logprobs', self.logprobs),
            max_tokens=kwargs.get('max_tokens', self.max_tokens),
        )
        messages = input['messages']
        output = self.model.chat(messages, sampling_params=sampling_params)
        response_objects = []
        for o in output[0].outputs:
            try:
                if response_object is not None:
                    response_objects.append(response_object.model_validate_json(o.text))
                else:
                    response_objects.append(o.text)
            except Exception as e:
                print(f"Error validating response: {e}")
                print(f"Response: {o.text}")
        return {
            'input': messages,
            'output': response_objects,
        }
    
    def batch_generate(
            self, 
            batch: List[List[Dict[str, str]]],
            response_object: BaseModel = None,
            **kwargs
            ) -> List[List[Dict[str, str]]]:
        """
        Generate a batch of responses from the vLLM model.
        """
        if response_object is not None:
            assert issubclass(response_object, BaseModel), "response_object must be a subclass of BaseModel"
            json_schema = response_object.model_json_schema()
            guided_decoding_params = GuidedDecodingParams(json=json_schema)
        else:
            guided_decoding_params = None
        sampling_params = SamplingParams(
            temperature=kwargs.get('temperature', self.temperature),
            guided_decoding=guided_decoding_params,
            top_p=kwargs.get('top_p', self.top_p),
            top_k=kwargs.get('top_k', self.top_k),
            repetition_penalty=kwargs.get('repetition_penalty', self.repetition_penalty),
            n=kwargs.get('n', self.num_samples),
            logprobs=kwargs.get('logprobs', self.logprobs),
            max_tokens=kwargs.get('max_tokens', self.max_tokens)
        )
        outputs = self.model.chat(batch, sampling_params=sampling_params)
        assert len(outputs) == len(batch), f"Expected {len(batch)} outputs, but got {len(outputs)}"
        responses = []
        for messages, output in zip(batch, outputs):
            response_objects = []
            for o in output.outputs:
                try:
                    if response_object is not None:
                        response_objects.append(response_object.model_validate_json(o.text))
                    else:
                        response_objects.append(o.text)
                except Exception as e:
                    print(f"Error validating response: {e}")
                    print(f"Response: {o.text}")
            responses.append({
                'input': messages,
                'output': response_objects,
            })
        return responses


class LLMAgent:
    def __init__(
            self,
            online_model_kwargs: Dict[str, Any] = None,
            offline_model_kwargs: Dict[str, Any] = None,
            use_cache: bool = True,
            cache_dir: str = 'cache/llm_agents',
            ):
        assert online_model_kwargs is not None or offline_model_kwargs is not None, "At least one of online_model_kwargs or offline_model_kwargs must be provided."
        if online_model_kwargs is not None:
            print("Using online model with OpenAI API")
            model_name = online_model_kwargs['model_name']
            self.agent = OpenAIClient(**online_model_kwargs)
        elif offline_model_kwargs is not None:
            model_name = offline_model_kwargs['model_name']
            agent_type = offline_model_kwargs.pop('agent_type', 'vllm')
            if agent_type == 'vllm':
                print("Using offline model with vLLM")
                self.agent = vLLMAgent(**offline_model_kwargs)
            elif agent_type == 'hf':
                print("Using offline model with Hugging Face Transformers")
                self.agent = HFAgent(**offline_model_kwargs)
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")
        self.use_cache = use_cache
        self.cache_dir = os.path.join(cache_dir, model_name)
        self.n = self.agent.num_samples
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

    def load_from_cache(self, cache_file: str, response_class, index) -> Dict[str, Any]:
        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            assert 'output' in data, "Cached response must contain 'output' key"
            response = {
                'index': index,
                'input': data['input'],
            }
            output = []
            for item in data['output']:
                if response_class is not None:
                    assert issubclass(response_class, BaseModel), "response_class must be a subclass of BaseModel"
                    item['output'] = response_class.model_validate_json(item['output'])
                output.append(item)
            response['output'] = output
        return response
    
    def save_to_cache(self, cache_file: str, response: Dict[str, Any]) -> None:
        """
        Save the response to cache.
        """
        with open(cache_file, 'w', encoding='utf-8') as f:
            to_save_response = copy.deepcopy(response)
            output = to_save_response.pop('output', None)
            serialized_output = []
            for item in output:
                if isinstance(item['output'], BaseModel):
                    item['output'] = item['output'].model_dump_json()
                serialized_output.append(item)
            to_save_response['output'] = serialized_output
            json.dump(to_save_response, f)

    def generate(self, input: Dict[str, Any], **kwargs) -> List[Dict[str, str]]:
        """
        Generate a response from the model.
        """
        use_cache = kwargs.pop('use_cache', self.use_cache)
        if not use_cache:
            return self.agent.generate(input, **kwargs)
        
        input_messages = input['messages']
        response_class = input.get('json_schema', None)
        if response_class is not None:
            assert issubclass(response_class, BaseModel), "response_class must be a subclass of BaseModel"
        input_str = f"messages: {input_messages}, json_schema: {response_class}, kwargs: {kwargs}"
        input_hash = sha256(input_str.encode('utf-8')).hexdigest()
        cache_file = f"{self.cache_dir}/{input_hash}.json"
        try:
            response = self.load_from_cache(cache_file, response_class, input['index'])
            # print(f"Loaded cached response for input hash {input_hash} from {cache_file}")
            return response
        except:
            response = self.agent.generate(input, **kwargs)
            # Save the response to cache
            self.save_to_cache(cache_file, response)
            return response
    
    def batch_generate(
            self, 
            batch: List[List[Dict[str, str]]],
            response_object: BaseModel = None,
            **kwargs
            ) -> List[List[Dict[str, str]]]:
        """
        Generate a batch of responses from the model.
        """
        batch_with_index = [{'index': i, 'messages': messages} for i, messages in enumerate(batch)]
        use_cache = kwargs.pop('use_cache', self.use_cache)
        if not use_cache:
            return self.agent.batch_generate(batch, response_object=response_object, **kwargs)
        
        responses = []
        to_compute = []
        for item in batch_with_index:
            input_str = f"messages: {item['messages']}, json_schema: {response_object}, kwargs: {kwargs}"
            input_hash = sha256(input_str.encode('utf-8')).hexdigest()
            cache_file = f"{self.cache_dir}/{input_hash}.json"
            try:
                cached_response = self.load_from_cache(cache_file, response_object, item['index'])
                # print(f"Loaded cached response for input hash {input_hash} from {cache_file}")
                responses.append(cached_response)
            except:
                item['hash'] = input_hash
                to_compute.append(item)
        if len(to_compute) == 0:
            return responses
        to_compute_batch = [item['messages'] for item in to_compute]
        to_compute_output = self.agent.batch_generate(to_compute_batch, response_object, **kwargs)
        # Save the responses to cache
        for item, output in zip(to_compute, to_compute_output):
            input_hash = item['hash']
            _output = copy.deepcopy(output)
            _output['index'] = item['index']
            responses.append(_output)
            cache_file = f"{self.cache_dir}/{input_hash}.json"
            self.save_to_cache(cache_file, output)

        # Sort responses by index to maintain order
        responses.sort(key=lambda x: x['index'])
        assert len(responses) == len(batch), f"Expected {len(batch)} responses, but got {len(responses)}"
        return responses


if __name__ == "__main__":
    # Example usage
    # Run on server
    # python -m sglang.launch_server --host 0.0.0.0 --model-path Qwen/Qwen3-8B --reasoning-parser qwen3 # --port 30000 
    online_model_kwargs = {
        'model_name': 'qwen3-32b',
        'url': 'http://n0998.talapas.uoregon.edu:30000/v1',
        'api_key': 'None',
        'concurrency': 64,
    }

    offline_model_kwargs = {
        'model_name': "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        'agent_type': 'vllm',
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

    online_model_kwargs.update(generate_kwargs)
    offline_model_kwargs.update(generate_kwargs)

    class CapitalResponse(BaseModel):
        capital: str
        country: str
        description: str

    queries = [
        [{'role': 'user', 'content': 'What is the capital of France? Please provide a JSON response'}],
        [{'role': 'user', 'content': 'What is the capital of Japan? Please provide a JSON response'}],
        [{'role': 'user', 'content': 'What is the capital of Germany? Please provide a JSON response'}],
    ]

    llm_agent = LLMAgent(online_model_kwargs=online_model_kwargs)
    response = llm_agent.generate({
        'messages': queries[0],
        'json_schema': CapitalResponse,
        'index': 0,
    })
    breakpoint()
    responses = llm_agent.batch_generate(queries, response_object=CapitalResponse)
    breakpoint()

