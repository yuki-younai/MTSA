from src.models.llm import LLM_local
from src.judge.llama3_guard_judge import Llama3_Guard_Judge
from src.reward.armreward import ArmoRMPipeline
from src.reward.cosine_sim import CosineSentenceEmbeddingReward
from src.reward.selfbule import SelfBleuReward
import argparse
import json
import random

def standardize(lst):
    if len(lst)==0:
        return lst
    mean = sum(lst) / len(lst)
    variance = sum((x - mean) ** 2 for x in lst) / len(lst)
    std_dev = variance ** 0.5
    standardized_lst = [(x - mean) / (std_dev+1e-6) for x in lst]
    return standardized_lst

def Alignment_idx_select(attack_results):
    revise_conv = []
    for idx,data in enumerate(attack_results):
        attack_sucess = data['attack_success']
        if not attack_sucess:
            continue
        
        llama_probs = data['llama_prob']
        random_idxs = []    
        for rand_idx, prob in enumerate(llama_probs):
            if prob<=0.3:
                random_idxs.append(rand_idx)
        if len(random_idxs)==0:
            continue
        alignment_idx = random.choice(random_idxs)
        data['revise_idx'] = alignment_idx#1
        revise_conv.append(data)
        
    return revise_conv

def red_team_revise(revise_conv):
    
    red_alignment_data = []
    repeat_num = 3
    temperature_list = [0.6, 0.8 ,1.0, 1.2]
    print("############################red_team_revise#########################################")
    for idx,data in enumerate(revise_conv):
        print(idx, " \ " , len(revise_conv))        
        alignment_idx= data['revise_idx']
        target_conv = data['target_conversation']
        attack_target = data['target']
        red_conv = data['attack_conversation']

        re_generate_conv = red_conv[:alignment_idx*2+1]
        new_attack_list = []
        
        for temper in temperature_list:
            temp_num = 0
            while True:
                temp_num += 1
                new_attack = attack_model.respond(re_generate_conv, max_tokens=400, temperature=temper)
                try:
                    think, question =  new_attack.split(f"Thinking {alignment_idx+1}:")[1].split(f"Attack question {alignment_idx+1}:")[0].strip(), new_attack.split(f"Attack question {alignment_idx+1}:")[1].strip()
                    break
                except:
                    if temp_num>repeat_num:
                        question = ' '
                        break      
            new_attack_conv = re_generate_conv.copy()
            new_attack_conv.append({"role":"user","content":new_attack})
            new_attack_list.append(new_attack_conv)

            
        temp = {}
        temp['new_red_conv_list'] = new_attack_list
        temp['origin_red_conv'] = red_conv
        temp['target_conv'] = target_conv
        temp['alignment_idx'] = alignment_idx
        temp['attack_target'] = attack_target
        red_alignment_data.append(temp)
        
    return red_alignment_data

def resample(red_data):
    
    repeat_num = 3
    print("############################tragectory resample#########################################")
    for idx,data in enumerate(red_data):
        print(idx,"\ ",len(red_data))
        alignment_idx = data['alignment_idx']
        new_red_conv_list = data['new_red_conv_list']
        target_conv = data['target_conv']
        new_red_conv_resample_list = []
        new_target_conv_resample_list = []
        erro = False
        for conv in new_red_conv_list:
            red_conv = conv.copy()
            new_target_conv = target_conv[:alignment_idx*2].copy()
            new_attack = red_conv[-1]['content']
            think, question =  new_attack.split(f"Thinking {alignment_idx+1}:")[1].split(f"Attack question {alignment_idx+1}:")[0].strip(), new_attack.split(f"Attack question {alignment_idx+1}:")[1].strip()
            new_target_conv.append({"role": "user", "content": question})
            target_response = target_model.respond(new_target_conv, temperature=0.7, max_tokens=200)
            new_target_conv.append({"role": "assistant", "content": target_response})
            red_conv.append({"role": "assistant", "content": target_response})
            
            for turn_idx in range(alignment_idx+1,5):
                temp_num = 0
                while True:
                    temp_num += 1
                    attack_response = attack_model.respond(red_conv, temperature=1.0, max_tokens=512)
                    try:
                        think, question =  attack_response.split(f"Thinking {turn_idx+1}:")[1].split(f"Attack question {turn_idx+1}:")[0].strip(), attack_response.split(f"Attack question {turn_idx+1}:")[1].strip()
                        break
                    except:
                        if temp_num>repeat_num:
                            erro = True
                            break
                        
                new_target_conv.append({"role": "user", "content": question})
                red_conv.append({"role": "user", "content": attack_response})
                target_response = target_model.respond(new_target_conv, temperature=1.0, max_tokens=400)
                new_target_conv.append({"role": "assistant", "content": target_response})
                red_conv.append({"role": "assistant", "content": target_response})
                
            new_red_conv_resample_list.append(red_conv)
            new_target_conv_resample_list.append(new_target_conv)
                
        if erro:
            print("error")
            continue
        data['new_red_conv_resample_list']  = new_red_conv_resample_list
        data['new_target_conv_resample_list'] = new_target_conv_resample_list       
        
    return red_data


def reward_judge(red_sample_data):

    output_data = []
    for data_idx, data in enumerate(red_sample_data):
        alignment_idx = data['alignment_idx']
        new_red_conv_resample_list = data['new_red_conv_resample_list']
        new_target_conv_resample_list = data['new_target_conv_resample_list']
        toxi_scores = []
        sim_scores = []
        for target_conv in new_target_conv_resample_list:
            toxi_prob = 0
            for idx in range(2*alignment_idx,9,2):
                messages = [{"role": "user", "content": target_conv[idx]}, {"role": "assistant", "content": target_conv[idx+1]}]
                safe_prob, unsafe_prob, judge = llama_judge_model.judge_prob(messages[-2:])
                toxi_prob += unsafe_prob
            toxi_scores.append(toxi_prob)
            
        toxi_scores = standardize(toxi_scores)

        cosine_scores = []
        selfblue_scores = []
        for red_conv in new_red_conv_resample_list:
            cosine_score = 0
            selfblue_score = 0
            for idx in range(2*alignment_idx,9,2):
                attack_prompt = red_conv[idx+1]['content']
                cosine_score += -cosine_model([attack_prompt])[0]
                selfblue_score += -selfblue_model([attack_prompt])[0]
            cosine_scores.append(cosine_score)
            selfblue_scores.append(selfblue_score)
            
        cosine_scores, selfblue_scores = standardize(cosine_scores), standardize(selfblue_scores)      
        sim_scores = [a + b for a, b in zip(cosine_scores, selfblue_scores)]
            
        result_scores = [a + b for a, b in zip(toxi_scores, sim_scores)]
        max_index = result_scores.index(max(result_scores))
        min_index = result_scores.index(min(result_scores))
        
        temp = {}
        temp['alignment_idx'] = alignment_idx
        temp['attack_target'] = data['attack_target']
        temp['origin_red_conv'] = data['origin_red_conv']
        temp['pos_red_conv'] = new_red_conv_resample_list[max_index]
        temp['neg_red_conv'] = new_red_conv_resample_list[min_index]
        temp['toxi_scores'] = toxi_scores
        temp['sim_scores'] = sim_scores
        temp['pos_index'] = max_index
        temp['neg_index'] = min_index
        output_data.append(temp)

    return output_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ActorAttack')
    parser.add_argument("--gpu", type=str, default='cuda:0', help="device id")
    parser.add_argument("--attack_model_name", type=str, default=None, help="Attack Model name.")
    parser.add_argument("--target_model_name", type=str, default=None, help="Target Model name.")
    parser.add_argument("--sim_model_name", type=str, default=None, help="Target Model name.")
    parser.add_argument("--judge_model_name", type=str, default=None, help="Target Model name.")
    parser.add_argument("--attack_results", type=str, default=None, help="Attack Data Path")
    parser.add_argument("--output_dir", type=str, default="Attack_results/", help="apply dynamic modification.")
    args = parser.parse_args()
    
    with open("datasets/red_team_data/red_team_data.json", 'r') as file:
        red_init_data = json.load(file)
    with open(args.attack_results, 'r') as file:
        attack_results = json.load(file)
    attack_results = attack_results
    cosine_model = CosineSentenceEmbeddingReward(args.sim_model_name, gpu=0)
    selfblue_model = SelfBleuReward()
    for jdx, ref in enumerate(red_init_data):
        ref_prompt = ''
        for conv in ref['red_team_conv']:
            if conv['role']=='user':
                ref_prompt += conv['content']
        cosine_model.append_reference(ref_prompt)
        selfblue_model.append_reference(ref_prompt)
        
    llama_judge_model = Llama3_Guard_Judge(model_name_path=args.judge_model_name, gpu=args.gpu)
    attack_model = LLM_local(model_name_or_path=args.attack_model_name, gpu=args.gpu, role='user')
    target_model = LLM_local(model_name_or_path=args.target_model_name, gpu=args.gpu, role='assistant')

    revise_conv = Alignment_idx_select(attack_results)
    red_alignment_data = red_team_revise(revise_conv)
    red_sample_data = resample(red_alignment_data)
    red_data_reward = reward_judge(red_sample_data)
    
    with open(args.output_dir, 'w') as file:  
        json.dump(red_data_reward, file, indent=4)

    








