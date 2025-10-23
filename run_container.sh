#!/bin/bash
podman run --rm -it --mount type=bind,src="$(pwd)",target=/workspace taylortorch