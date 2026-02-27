from ultralytics import YOLO
import torch

model = YOLO("yolov6.pt")
model.to("cuda")
model.eval()

dummy = torch.randn(1, 3, 640, 512).cuda()

for _ in range(50):
    _ = model.model(dummy)

starter = torch.cuda.Event(enable_timing=True)
ender = torch.cuda.Event(enable_timing=True)

times = []
for _ in range(200):
    starter.record()
    _ = model.model(dummy)
    ender.record()
    torch.cuda.synchronize()
    times.append(starter.elapsed_time(ender))

print("Latency:", sum(times)/len(times), "ms")