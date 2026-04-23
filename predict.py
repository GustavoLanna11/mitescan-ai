import torch
from PIL import Image
from torchvision import transforms
from model.cnn import MiteScanCNN

classes = ["normal", "varroa", "deformada"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

model = MiteScanCNN()

# força criação da fc1
dummy_input = torch.randn(1, 3, 224, 224)
model(dummy_input)

model.load_state_dict(torch.load("model.pth"))
model.eval()

img = Image.open("teste.jpg").convert("RGB")
img = transform(img).unsqueeze(0)

with torch.no_grad():
    output = model(img)
    probs = torch.softmax(output, dim=1)

    # 👇 ADICIONA AQUI
    for i, prob in enumerate(probs[0]):
        print(classes[i], ":", round(prob.item(), 2))

    classe_idx = torch.argmax(probs).item()
    confianca = probs[0][classe_idx].item()

print("\nClasse final:", classes[classe_idx])
print("Confiança:", round(confianca, 2))