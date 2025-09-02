# delete old images by tag
docker rmi contanos:base-opencv-cpu contanos:base-onnx-gpu contanos:base-pytorch-gpu

# build new images
docker build stride/base-opencv-cpu -t contanos:base-opencv-cpu --no-cache
docker build stride/base-onnx-gpu -t contanos:base-onnx-gpu --no-cache
docker build stride/base-pytorch-gpu -t contanos:base-pytorch-gpu --no-cache
