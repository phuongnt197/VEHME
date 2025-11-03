# VEHME:  Vision-Language Model For Evaluating Handwritten Mathematics Expressions
<sub> **Thu Phuong Nguyen***, **Duc M. Nguyen***, Hyotaek Jeon, Hyunwook Lee, Hyunmin Song, *Sungahn Ko***, and *Taehwan Kim*** </sub>

<p align="right"> <sub>* Equal contribution <br> ** Co-corresponding authors </sub> </p>

---

This is an official implementation of VEHME in the following paper: VEHME:  Vision-Language Model For Evaluating Handwritten Mathematics Expressions, EMNLP 2025 (Main Conference)


## Dependencies
You can create a new virtual environment and install all of the dependencies via the following command
```bash
create --name vehme --file environment.yml
```

## Data
* Please download the AIHub data [here](https://www.aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&searchKeyword=%EC%88%98%ED%95%99&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&aihubDataSe=data&dataSetSn=71716)
* Please refer to [AI4Bharat/FERMAT](https://github.com/AI4Bharat/FERMAT) repository to download the FERMAT dataset.

## Expression-aware Visual Prompting Module
### Multi-Line Expression Canvas Synthesis
* Clone [thanhnghiadk/syntactic_HME_generation](https://github.com/thanhnghiadk/syntactic_HME_generation) repository to generate syntactically valid handwritten mathematical expression patterns from CROHME 2024 training set.
```bash
cd src/evpm
git clone https://github.com/thanhnghiadk/syntactic_HME_generation.git
cd syntactic_HME_generation
```
* Run the following command to generate **single-line mathematical expressions** dataset. If it does not work, please follow the instruction in the repository for more details
```bash
python syntactic_data_generation.py
```
### Training EVPM
Run the following script to train the EVPM:
```bash
python train.py
```
Warning: This script is intended to use with GPU. If you train EVPM with CPU, it is going to take a long time.

## Stage 1: SFT
To warm-up with sft, modify and run the following script:
```bash
cd src
sh scripts/sft.sh
```
Make sure to set up the data before training.

## Stage 2: GRPO
Warning: This stage utilize multiple GPUs! Make sure to have at least 1 GPU for the QwQ as the localization reward model, and the rest for training VEHME.
### Deploying QwQ with vllm
We host [QwQ-32B](https://huggingface.co/Qwen/QwQ-32B) locally with [vllm](https://github.com/vllm-project/vllm) via the following script:
```bash
cd src
sh scripts/serve_rewards.sh
```

### Finetuning with GRPO
To train the second stage using Group Relative Preference Optimization (GRPO), modify and run the following script:
```bash
cd src
sh scripts/grpo.sh
```

## Inference
To evaluate VEHME or any open-source models, please follow the following step:

* Deploy the model with vllm (make sure to update the name of the model):
```bash
cd src
sh scripts/serve_vllm.sh
```

* Modify and run the following scripts:
```bash
python -m instr_tuning.eval.vlm_eval --base_url [vllm url] --port [port number] --model [moden name] --dataset [path/to/dataset] --num_workers [number of concurent requests] 
python -m instr_tuning.eval.evaluate_inference_result --base_url [vllm url] --port [port number] --model [moden name] --num_workers [number of concurent requests] --eval_localization True
```
The result can be found in directory `src/instr_tuning/eval/results`.

## Acknowledgement
Our SFT and GRPO implementation is based on the [`ms-swift` framework](https://github.com/modelscope/ms-swift), and [`vllm` framework](https://github.com/vllm-project/vllm) for efficient inference. We sincerely thank all contributors for their amazing work.

## Citation
```bibtex
@inproceedings{nguyen-etal-2025-vehme,
    title = "{VEHME}: A Vision-Language Model For Evaluating Handwritten Mathematics Expressions",
    author = "Nguyen, Thu Phuong  and
      Nguyen, Duc M.  and
      Jeon, Hyotaek  and
      Lee, Hyunwook  and
      Song, Hyunmin  and
      Ko, Sungahn  and
      Kim, Taehwan",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.1619/",
    pages = "31781--31801",
    ISBN = "979-8-89176-332-6",
    abstract = "Automatically assessing handwritten mathematical solutions is an important problem in educational technology with practical applications, but remains a significant challenge due to the diverse formats, unstructured layouts, and symbolic complexity of student work. To address this challenge, we introduce VEHME-a $\textbf{V}$ision-Language Model for $\textbf{E}$valuating $\textbf{H}$andwritten $\textbf{M}$athematics $\textbf{E}$xpressions{---}designed to assess open-form handwritten math responses with high accuracy and interpretable reasoning traces. VEHME integrates a two-phase training pipeline: (i) supervised fine-tuning using structured reasoning data, and (ii) reinforcement learning that aligns model outputs with multi-dimensional grading objectives, including correctness, reasoning depth, and error localization. To enhance spatial understanding, we propose an Expression-Aware Visual Prompting Module, trained on our synthesized multi-line math expressions dataset to robustly guide attention in visually heterogeneous inputs. Evaluated on AIHub and FERMAT datasets, VEHME achieves state-of-the-art performance among open-source models and approaches the accuracy of proprietary systems, demonstrating its potential as a scalable and accessible tool for automated math assessment. Our training and experiment code is publicly available at our GitHub repository."
}
```
