from utils import *
import io
import base64

labels = ["Normal", "Diabetic Retinopathy"]
model = load_model()
model.eval()

# val_auc_dr = 0.9808 DR for 120 fundus image


def main(img_path, idx):
    img = Image.open(img_path)
    torch_img, normed_torch_img = preprocess(img)
    DR, _ = get_probas(normed_torch_img, model)
    heatmap = get_heatmap(torch_img, normed_torch_img,
                          model, idx, DR[idx])
    plt.imshow(heatmap)
    plt.show()


def handler(event, context):
    base64_string = event["image"]
    img = stringToRGB(base64_string)
    torch_img, normed_torch_img = preprocess(img)
    DR, _ = get_probas(normed_torch_img, model)
    heatmaps = []
    for i in range(len(labels)):
        heatmap = get_heatmap(torch_img, normed_torch_img, model, i, DR[i])
        heatmap = RGBToString(heatmap)
        heatmaps.append(heatmap)
    return {"labels": labels, "probas": DR, "heatmap": heatmaps}


if __name__ == "__main__":
    img_path = "test.tif"
    idx = 1
    main(img_path, idx)
