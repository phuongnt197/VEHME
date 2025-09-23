# VEHME:  Vision-Language Model For Evaluating Handwritten Mathematics Expressions
<small> **Thu Phuong Nguyen***, **Duc M. Nguyen***, Hyotaek Jeon, Hyunwook Lee, Hyunmin Song, *Sungahn Ko***, and *Taehwan Kim*** </small>

<div style="text-align: right"> * Equal contribution <br> ** Co-corresponding authors </div>

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
@inproceedings{nguyen2025vehme,
    title={{VEHME}: A Vision-Language Model For Evaluating Handwritten Mathematics Expressions},
    author={Thu Phuong Nguyen and Duc M. Nguyen and Hyotaek Jeon and Hyunwook Lee and Hyunmin Song and Sungahn Ko and Taehwan Kim},
    booktitle={The 2025 Conference on Empirical Methods in Natural Language Processing},
    year={2025},
    url={https://openreview.net/forum?id=lLlIXm4KNE}
}
```
