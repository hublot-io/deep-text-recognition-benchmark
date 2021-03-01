from model import Model
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model.load_from_checkpoint("models/ocr_128_512.ckpt")
# model = Model()

sampleImg = torch.rand(1, 1, 32, 100).to(device)
sampleTxt = torch.full((1, 26), 0).to(device)
script = model.to_torchscript()
# torch.jit.save(script, 'model.pt')
# torch.jit.save(script, 'model.pt', method="trace",
#                example_inputs=[(sampleImg, sampleTxt)])

# model.to_onnx("model.onnx")
