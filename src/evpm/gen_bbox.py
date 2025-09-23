import random
from utils import create_canvas
from tqdm import tqdm
import glob

synthetic_crohme_data = glob.glob('syntactic_HME_generation/semantic_dataset/*.inkml')
synthetic_crohme_data = sorted(synthetic_crohme_data)
len(synthetic_crohme_data)

if __name__ == "__main__":

    # Generate training data
    for iter in tqdm(range(10000)):
        while True:
            try:
                number_of_hme = random.randint(1, 10)
                hmes = random.choices(synthetic_crohme_data, k=number_of_hme)
                canvas, list_bbox = create_canvas(hmes)
                canvas.save(f"./data/images/train/{iter}.png")
                with open(f"./data/labels/train/{iter}.txt", "w") as f:
                    for bbox in list_bbox:
                        f.write("0 ")
                        for x, y in bbox:
                            f.write(f"{x.item() / canvas.width} {y.item() / canvas.height} ")
                        f.write("\n")
                break
            except Exception as e:
                continue

    # Generate validation data
    for iter in tqdm(range(1000)):
        while True:
            try:
                number_of_hme = random.randint(1, 10)
                hmes = random.choices(synthetic_crohme_data, k=number_of_hme)
                canvas, list_bbox = create_canvas(hmes)
                canvas.save(f"/data/labels/images/val/{iter}.png")
                with open(f"/data/labels/labels/val/{iter}.txt", "w") as f:
                    for bbox in list_bbox:
                        f.write("0 ")
                        for x, y in bbox:
                            f.write(f"{x.item() / canvas.width} {y.item() / canvas.height} ")
                        f.write("\n")
                break
            except Exception as e:
                continue
