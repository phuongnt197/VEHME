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

## Training EVPM
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

## Stage 1: SFT


## Stage 2: GRPO


## Inference


## Acknowledgement


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
