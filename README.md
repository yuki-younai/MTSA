# MTSA

The official implementation of our paper "[MTSA: Multi-turn Safety Alignment for LLMs through Multi-round Red-teaming](https://arxiv.org/abs/2505.17147)

![Safety Alignment](https://img.shields.io/badge/Jailbreak-Attacks-yellow.svg?style=plastic)
![Adversarial Attacks](https://img.shields.io/badge/Adversarial-Attacks-orange.svg?style=plastic)
![Large Language Models](https://img.shields.io/badge/LargeLanguage-Models-green.svg?style=plastic)
---

## 📚 Abstract

The proliferation of jailbreak attacks against large language models (LLMs) highlights the need for robust security measures. However, in multi-round dialogues, malicious intentions may be hidden in interactions, leading LLMs to be more prone to produce harmful responses. In this paper, we propose the Multi-Turn Safety Alignment (MTSA) framework, to address the challenge of securing LLMs in multi-round interactions. It consists of two stages: In the thought-guided attack learning stage, the red-team model learns about thought-guided multi-round jailbreak attacks to generate adversarial prompts. In the adversarial iterative optimization stage, the red-team model and the target model continuously improve their respective capabilities in interaction. Furthermore, we introduce a multi-turn reinforcement learning algorithm based on future rewards to enhance the robustness of safety alignment. Experimental results show that the red-team model exhibits state-of-the-art attack capabilities, while the target model significantly improves its performance on safety benchmarks.

![pipeline](figures/framework.png)

### TODO List  

- [] Code Implementation
- [] Public the Model

## 🚀 Quick Start

- **Get code**

```shell 
git clone https://github.com/yuki-younai/MTSA.git
```

- **Build environment**

```shell
cd MTSA
conda create -n mtsa python==3.11
conda activate mtsa
pip install -r requirements.txt
```
- **Download Models**

```shell
Download Qwen2.5-7B-Instruct [link](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
Download Llama-Guard-3-8B [link](https://huggingface.co/meta-llama/Llama-Guard-3-8B)
Download zephyr-7b-beta [link](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta)
Download all-MiniLM-L6-v2  [link](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
Download ArmoRM-Llama3-8B-v0.1 [link](https://huggingface.co/RLHFlow/ArmoRM-Llama3-8B-v0.1)
```

- **Red-team model Initialize**

```shell
bash script/red_team_sft.sh \
$Qwen25_7B_Instruct \
datasets/red_team_data 
```
- **Adversarial Generate**

```python
python adversarial_generate.py --gpu cuda:0 \
        --attack_model_name $Red_team_model_path \
        --judge_model_name $Llama-Guard-3-8B \
        --target_model_name $Target_model_path \
        --attack_data_path datasets/attack_target/train_attack_target.json \
        --output_dir attack_results/adversarial_generate.json
```

- **Red-team model Optimize**

1.Generate the trajectory resampling data for the red team model
```python
python red_team_tragectory_resample.py --gpu cuda:0 \
        --attack_model_name $Red_team_model_path \
        --judge_model_name $Llama-Guard-3-8B \
        --target_model_name $Target_model_path \
        --sim_model_name $all-MiniLM-L6-v2 \
        --attack_results attack_results/adversarial_generate.json \
        --output_dir attack_results/red_team_dpo/red_team_tragectory_resample.json
```
2.Perform preference optimization on the red team model
```bash
bash script/red_team_dpo.sh \
$Red_team_model_path \
attack_results/red_team_dpo
```

- **Target model Optimize**

1.Generate the trajectory resampling data for the target model
```python
python target_tragectory_resample.py --gpu cuda:0 \
        --attack_model_name $Red_team_model_path \
        --judge_model_name $Llama-Guard-3-8B \
        --target_model_name $Target_model_path \
        --reward_model_name $ArmoRM-Llama3-8B-v0.1 \
        --attack_results attack_results/adversarial_generate.json \
        --output_dir attack_results/safe_align/target_tragectory_resample.json
```
2.Perform preference optimization on the target model
```bash
bash script/safe_mtrlhf.sh \
$Target_model_path \
attack_results/safe_align
```

## 📎 Reference BibTeX

```bibtex
@article{guo2024mtsa,
      title={MTSA: Multi-turn Safety Alignment for LLMs through Multi-round Red-teaming},
      author={Weiyang Guo and Jing Li and Wenya Wang and YU LI and Daojing He and Jun Yu and Min Zhang},
      journal={arXiv preprint arXiv:2505.17147},
      year={2025},
      url={https://arxiv.org/abs/2505.17147}
}
```


