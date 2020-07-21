
## Use the container (with docker ≥ 19.03)

```
cd docker/
# Build:
docker build -t py-bottom-up-attention .
# Run:
docker run --gpus all -it \
	--shm-size=8gb --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	--name=py-bottom-up-attention \
    --volume="<images_folder>:/images:rw" \
    --volume=$HOME/.torch/fvcore_cache:/tmp:rw

```

Replace `<image_folder>` with the folder on your workstation that contains the images that have to be processed.
