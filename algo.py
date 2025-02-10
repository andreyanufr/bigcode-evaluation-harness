import torch
from transformers import AutoModelForCausalLM
from abc import ABC
from abc import abstractmethod
from typing import Dict, List, Union
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
import torch.nn.functional as F

class GenAlgo(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def generate(self, *args, **kwargs):
        pass


class ArmoRMPipeline:
    def __init__(self, model_id, device_map="auto", torch_dtype=torch.bfloat16, truncation=True, trust_remote_code=False, max_length=4096):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
        )
        self.truncation = truncation
        self.device = self.model.device
        self.max_length = max_length

    def process(self, messages: List[Dict[str, str]]) -> Dict[str, float]:
        """
        messages: OpenAI chat messages to be scored
        Note: no batching since due to length differences, the model will have to pad to the max length which is not efficient
        Returns: a dictionary with the score between 0 and 1
        """
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            padding=True,
            truncation=self.truncation,
            max_length=self.max_length,
        ).to(self.device)
        with torch.no_grad():
            output = self.model(input_ids)
            try:
                score = output.score.float().item()
            except:
                score = list(output.score.float().cpu().numpy())
        return {"score": score}
    
    def __call__(self, prompt: Union[str, List[Dict[str, str]]], responses: List[str]) -> Dict[str, float]:
         reward_inputs = [[{"role": "user", "content": prompt}, {"role": "assistant", "content": t}] for t in responses]
         return self.process(reward_inputs)


class DecisionTreeRMPipeline:
    def __init__(self, model_id="RLHFlow/Decision-Tree-Reward-Llama-3.1-8B", device_map="cuda", torch_dtype=torch.bfloat16, truncation=True, trust_remote_code=True, max_length=4096):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        self.model = self.model.to('cuda:1')
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
        )
        self.truncation = truncation
        self.device = self.model.device
        self.max_length = max_length

    @staticmethod
    def convert_to_chat_format(prompt, response=None):
        if "<extra_id_1>" in prompt:
            """
            Handling HelpSteer2 prompts which may contain multi-turn conversations with the special token <extra_id_1>
            """
            turns = prompt.split("<extra_id_1>")
            conversation = []
            conversation.append({
                "role": "user",
                "content": turns[0]
            })
            
            for i in range(1, len(turns)):
                parts = turns[i].split("\n", 1)
                role = parts[0]
                content = parts[1]
                conversation.append({
                    "role": "assistant" if role == "Assistant" else "user",
                    "content": content
                })
        else:
            conversation = [{"role": "user", "content": '# You are professional software engineer and mathematician. ' + prompt}]
        if response is not None:
            conversation.append({"role": "assistant", "content": response})
        return conversation

    @staticmethod
    def process_conversation(conversation):
        for message in conversation:
            message["content"] = message["content"].rstrip('\n')
        return conversation

    @torch.no_grad()
    def compare_batch(self, prompt: Union[str, List[Dict[str, str]]], responses: List[str]):
        """
        Compare two inputs and return the difference in scores
        """
        if isinstance(prompt, str):
            conversation = self.convert_to_chat_format(prompt)
        elif isinstance(prompt, list):
            conversation = prompt
        else:
            raise ValueError(f"The prompt must be a string or a list of dictionaries, but got {type(prompt)}")
        assert isinstance(conversation, list), "The conversation must be a list of dictionaries"
        assert len(conversation) >= 1, "The conversation must have at least one message (as prompt)"
        assert conversation[-1]["role"] == "user", "The last message in the conversation must be from the user"
        conversations = [conversation + [{"role": "assistant", "content": response}] for response in responses]
        conversations = [self.process_conversation(conversation) for conversation in conversations]

        convs_tokenized = self.tokenizer.apply_chat_template(conversations, tokenize=True, return_tensors="pt", truncation='longest_first', padding=True).to(self.model.device)
        while 128009 in convs_tokenized[:, -1]:
            convs_tokenized = convs_tokenized[:, :-1]

        embeddings = self.model.forward(convs_tokenized, output_hidden_states=True).hidden_states[-1][:,-1]#.float().cpu().numpy()
        
        # weight = self.model.score.weight.float().cpu().numpy()
        # bias = self.model.score.bias.float().cpu().numpy()
        # rewards = embeddings @ weight.T + bias
        
        rewards = self.model.score(embeddings).float().cpu().numpy()

        return {
            "rewards": rewards,
            "attributes": self.model.attributes
            }
    
    def __call__(self, prompt: Union[str, List[Dict[str, str]]], responses: List[str]):
        res = self.compare_batch(prompt, responses)
        res = {'score': res['rewards'][:, 0] + res['rewards'][:, 1]} # helpfulness + correctness
        #res = {'score': res['rewards'][:, 3]} # complexity
        return res


class TransitionsReward:
    def __init__(self, model_id="Qwen/Qwen2.5-Coder-7B-Instruct", device_map="cuda", torch_dtype=torch.bfloat16, truncation=True, trust_remote_code=True):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            #load_in_8bit=True,
        )
        self.model = self.model.to('cuda:1')
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
        )
        self.truncation = truncation
        self.device = self.model.device

    @staticmethod
    def make_chat_prompt(prompt: str, response: str, tokenizer: AutoTokenizer) -> str:
        # directly return prompt if it does not have a tokenizer.chat_template
        if tokenizer.chat_template is None:
            return prompt

        response = f"""\
        Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:
        ```python
        {response}
        ```
        """
        prompt = [
                {
                    "role": "user",
                    "content": prompt
                },
                {
                    "role": "assistant",
                    "content": response
                },
            ]

        return prompt

    @staticmethod
    def process_conversation(conversation):
        for message in conversation:
            message["content"] = message["content"].rstrip('\n')
        return conversation

    @torch.no_grad()
    def compare_batch(self, prompt: Union[str, List[Dict[str, str]]], responses: List[str]):
        """
        Compare two inputs and return the difference in scores
        """
        conversations = [self.make_chat_prompt(prompt, response, self.tokenizer) for response in responses]

        convs_tokenized = self.tokenizer.apply_chat_template(conversations, tokenize=True, return_tensors="pt", truncation='longest_first', padding=True).to(self.model.device)
        
        start = 0
        # while 151643 in convs_tokenized[:, -1]:
        #     convs_tokenized = convs_tokenized[:, :-1]

        while torch.all(convs_tokenized[:, start] == convs_tokenized[0, start]):
            start += 1

        logits = self.model(convs_tokenized).logits
        
        transition_scores = []
        for i in range(len(responses)):
            idxs = convs_tokenized[i]
            log_probs = torch.nn.functional.softmax(logits[i, ...], dim=-1)#[:-1, :]
            score = [log_probs[j, idxs[j]].item() for j in range(idxs.shape[0]) if idxs[j] != 151643]
            score = score[start:]
            score = sorted(score)
            #score = score[:len(score) // 2]
            score = torch.mean(torch.tensor(score))
            transition_scores.append(score)
        transition_scores = torch.tensor(transition_scores)        

        return transition_scores
    
    def __call__(self, prompt: Union[str, List[Dict[str, str]]], responses: List[str]):
        res = self.compare_batch(prompt, responses)
        res = {'score': res.squeeze().cpu().numpy()}
        return res


class MathPRMPIpeline:
    def __init__(self):
        model_name = "Qwen/Qwen2.5-Math-PRM-7B"
        device = "auto"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name, 
            device_map=device, 
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval()
    
    @staticmethod
    def make_step_rewards(logits, token_masks):
        probabilities = F.softmax(logits, dim=-1)
        probabilities = probabilities * token_masks.unsqueeze(-1) # bs, seq_len, num_labels
        
        all_scores_res = []
        for i in range(probabilities.size(0)):
            sample = probabilities[i] # seq_len, num_labels
            positive_probs = sample[sample != 0].view(-1, 2)[:, 1] # valid_tokens, num_labels
            non_zero_elements_list = positive_probs.cpu().tolist()
            all_scores_res.append(non_zero_elements_list)
        return all_scores_res


    def compute_reward(self, prompt, response):
        data = {
            "system": "Please reason step by step, and solve task using python language.",
            "query": prompt,
            "response": [
                response
            ]
        }

        messages = [
            {"role": "system", "content": data['system']},
            {"role": "user", "content": data['query']},
            {"role": "assistant", "content": "<extra_0>".join(data['response']) + "<extra_0>"},
        ]
        conversation_str = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )

        input_ids = self.tokenizer.encode(
            conversation_str, 
            return_tensors="pt", 
        ).to(self.model.device)

        outputs = self.model(input_ids=input_ids)

        step_sep_id = self.tokenizer.encode("<extra_0>")[0]
        token_masks = (input_ids == step_sep_id)
        step_reward = self.make_step_rewards(outputs[0], token_masks)
        return step_reward
    
    @torch.no_grad()
    def compare_batch(self, prompt: Union[str, List[Dict[str, str]]], responses: List[str]):
        res = []
        for response in responses:
            res.append(self.compute_reward(prompt, response))
        
        res = {'score': res}
        return res
    
    def __call__(self, prompt: Union[str, List[Dict[str, str]]], responses: List[str]):
        res = []
        for response in responses:
            res.append(self.compute_reward(prompt, response))
        
        res = {'score': res}
        return res


class BestOfN(GenAlgo):
    def __init__(self,
                 model: AutoModelForCausalLM,
                 tokenizer: AutoTokenizer,
                 reward_model: ArmoRMPipeline,
                 n_samples=28,
                 iters=3,
                 max_tokens_in_iter=128):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.reward_model = reward_model
        self.n_samples = n_samples
        self.iters = iters
        self.max_tokens_in_iter = max_tokens_in_iter
        self.split = 3

    def generate_(self,
                input_ids,
                num_return_sequences,
                **gen_kwargs):
        if not 'min_p' in gen_kwargs:
            gen_kwargs['min_p'] = 0.25
        prompt = self.tokenizer.decode(input_ids[0])
        prompt_len = input_ids.shape[1]
        #input_ids = input_ids['input_ids']
        
        input_ids = input_ids.repeat(self.n_samples, 1)
        iter_repeats = self.n_samples // self.split
    
        prev_len = prompt_len
        early_out = False
        best_results = []
        eot = self.tokenizer.added_tokens_encoder['<|endoftext|>']
        split = self.split
        
        # bad_scores = [
        #    # "Write a function to find sequences of lowercase letters joined with an underscore.",
        #    # "Write a function to find sequences of lowercase letters joined with an underscore.",
        #     "Write a function to check if the given number is woodball or not."
        # ]
        # is_bs = False
        # for bs in bad_scores:
        #     if bs in prompt:
        #         print("1")
        #         is_bs = True
        #         break

        for ii in range(self.iters):
            if split == 0:
                break
            result = self.model.generate(input_ids, **gen_kwargs, max_new_tokens=self.max_tokens_in_iter)
            # if result.shape[1] < prev_len + self.max_tokens_in_iter:
            #     early_out = True
            #     break
            texts = self.tokenizer.batch_decode(result[:, prompt_len: ], skip_special_tokens=True)
            scores = []
            
            texts = ["\n".join(t.split('\n')[:-1]) for t in texts]
           
            score = self.reward_model(prompt, texts)
            for i in range(len(texts)):
                scores.append([score['score'][i], i])
            
            scores = sorted(scores, key=lambda x: x[0], reverse=True)
            # if is_bs:    
            #     print("Iteration: ", ii, prompt)
            #     for i in range(len(scores)):
            #         print('Score ', scores[i][0])
            #         print(texts[scores[i][1]])
            #         print("########################")
            #     print("?????????????????????????????????????????????????????????????\n\n")
            
            if ii + 1 < self.iters:
                cur_results = []
                for i in range(split):
                    if result[scores[i][1], -1].item() == eot:
                        tmp = result[scores[i][1], :]
                        sz = tmp.shape[0]
                        while sz > 1 and tmp[sz - 1] == eot:
                            sz = sz - 1
                        best_results.append(tmp[:sz])
                        split = split - 1
                    else:
                        cur_results.append(result[scores[i][1], :].repeat(iter_repeats, 1))
                if len(cur_results) == 0:
                    break
                input_ids = torch.concat(cur_results, 0)
                if (len(input_ids.shape) == 1):
                    input_ids = input_ids.unsqueeze(0)
                prev_len = result.shape[1]
            elif len(best_results) < self.n_samples:
                for i in range(result.shape[0]):
                    if len(best_results) == self.n_samples:
                        break
                    tmp = result[scores[i][1], :]
                    sz = tmp.shape[0]
                    while sz > 1 and tmp[sz - 1] == eot:
                        sz = sz - 1
                    best_results.append(tmp[:sz])
                    
    
        if early_out or True:
            #texts = self.tokenizer.batch_decode(best_results[:, prompt_len: ], skip_special_tokens=True)
            texts = [self.tokenizer.decode(best_results[i][prompt_len:], skip_special_tokens=True) for i in range(len(best_results))]
            scores = []
            
            # for i, t in enumerate(texts):
            #     score = self.reward_model([{"role": "user", "content": prompt}, {"role": "assistant", "content": t}])
            #     scores.append([score['score'], i])
        
            score = self.reward_model(prompt, texts)
            for i in range(len(texts)):
                scores.append([score['score'][i], i])
            
            scores = sorted(scores, key=lambda x: x[0], reverse=True)
        
        #texts = self.tokenizer.batch_decode(result[:, prompt_len: ])
        print(prompt)
        for i in range(1):#self.n_samples):
            print('Score ', scores[i][0])
            print(texts[scores[i][1]])
            print("########################")
        print("\n\n")

        best_results = [best_results[scores[i][1]] for i in range(num_return_sequences)]
        result = best_results[0]#torch.concat(best_results, 0)
        
        if (len(result.shape) == 1):
            result = result.unsqueeze(0)
        
        return result

    @staticmethod
    def clean_text(t):
        tmp = t.split('\n')
        if 'return' in tmp[-1]:
            return t
        # sz = len(tmp) - 1
        # while sz > 0:
        #     if '# Test the function' in tmp[sz]:
        #         break
        #     sz -= 1
        # if sz > 0:
        #     tmp = tmp[:sz]
        tmp = list(filter(lambda x : len(x) > 1, tmp))
        tmp = tmp[:-1]
        return "\n".join(tmp)


    def generate(self,
                input_ids,
                num_return_sequences,
                **gen_kwargs):
        if not 'min_p' in gen_kwargs:
            gen_kwargs['min_p'] = 0.35
        gen_kwargs['num_return_sequences'] = self.n_samples
        
        prompt = self.tokenizer.decode(input_ids[0])
        prompt_len = input_ids.shape[1]
        #input_ids = input_ids['input_ids']
        
        #input_ids = input_ids.repeat(self.n_samples, 1)
        result = self.model.generate(input_ids, **gen_kwargs)
        texts = self.tokenizer.batch_decode(result[:, prompt_len: ], skip_special_tokens=True)
        
        if 'joined with an underscore' in prompt:
            print("")
        # avoid non complited strings
        texts = [self.clean_text(t) for t in texts]
        
        scores = []
        score = self.reward_model(prompt, texts)
        for i, t in enumerate(texts):
            if 'return ' in t:
                scores.append([score['score'][i], i])
            else:
                scores.append([0.1 * score['score'][i], i])
        
        scores = sorted(scores, key=lambda x: x[0], reverse=True)
        
        #texts = self.tokenizer.batch_decode(result[:, prompt_len: ])
        print(prompt)
        for i in range(1):#self.n_samples):
            print('Score ', scores[i][0])
            print(texts[scores[i][1]])
            print("########################")
        print("\n\n")

        #num_return_sequences = 10
        best_results = [result[scores[i][1]] for i in range(num_return_sequences)]
        result = best_results[0]#torch.vstack(best_results) # torch.concat(best_results, 0)
        
        if (len(result.shape) == 1):
            result = result.unsqueeze(0)
        
        return result


if __name__ == "__main__":
    model_id = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    device_map = "auto"
    torch_dtype = torch.bfloat16
    truncation = True
    trust_remote_code = False
    max_length = 4096

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    rm = ArmoRMPipeline("RLHFlow/ArmoRM-Llama3-8B-v0.1", trust_remote_code=True)
    
    bon = BestOfN(model, tokenizer, rm, iters=10)
    
    
    prompt = '"""Write a python function to remove first and last occurrence of a given character from the string.\n'
    
    input_ids = tokenizer(
            prompt,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
    gen = bon.generate(input_ids, 1, temperature=0.8, min_p=0.25, top_p=0.9)
    
    
    print(prompt)
    
    print(tokenizer.decode(gen))
        
        
        
        
        
        