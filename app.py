from flask import Flask, request, jsonify, send_file
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
import os

from model import AttributeCNN
from cosine_similarity import top_k

model = AttributeCNN(num_classes=40)
model.load_state_dict(torch.load("models/Model3.pth", map_location='cpu'))
device= torch.device("cuda")
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

app = Flask(__name__)

thresholds = torch.load('optimal__thresholds.pt')
thresholds = torch.tensor(thresholds).unsqueeze(0)

@app.route('/predict', methods=['POST'])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image Uploaded"}, 400)
    files = request.files['image']
    image = Image.open(io.BytesIO(files.read())).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(next(model.parameters()).device)

    with torch.no_grad():
        output = model(image_tensor)


        preds = (output > thresholds).int()
        predictions = preds.squeeze(0).tolist()

        similar = top_k(predictions, 1)['filename'].values[0]
        print
        image_path = os.path.join("/home/skrubstar/datasets/celeba/img_celeba/img_celeba", similar)

    return send_file(image_path, mimetype='image/jpeg')

if __name__ == "__main__":
    app.run(debug=True)

