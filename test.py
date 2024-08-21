import torch
from ldm.models.autoencoder import AutoencoderKL
from ldm.data.dark_zurich import DarkZurichNightTrain, DarkZurichDayTrain
import yaml
import cv2
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

def test_autoencoder(device):
    print("Testing autoencoder...")
    config = yaml.safe_load(open("models/first_stage_models/kl-f4/config.yaml"))[
        "model"
    ]["params"]
    model = AutoencoderKL(
        config["ddconfig"], config["lossconfig"], config["embed_dim"], "models/first_stage_models/kl-f4/model.ckpt"
    )
    model = model.to(device)
    model.eval()

    image = cv2.resize(cv2.imread("image.jpeg") / 255, (256, 256))
    image = torch.from_numpy(image).unsqueeze(0).float().permute(0, 3, 1, 2)
    out, _ = model(image.to(device))
    out = out.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    cv2.imwrite("ae.jpg", out * 255)

def test_data():
    print("Testing training dataset...")
    dset = DarkZurichDayTrain(size=256)
    loader = DataLoader(dset, batch_size=8)
    batch = next(iter(loader))['image'].permute(0, 3, 1, 2)
    save_image(make_grid(batch), "data.png")




if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_data()
