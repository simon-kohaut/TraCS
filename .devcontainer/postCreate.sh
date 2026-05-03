#!/bin/bash

sudo apt-get update && sudo apt-get install -y gringo && export CLINGO_LIBRARY_PATH=/usr/lib/$(dpkg-architecture -qDEB_HOST_MULTIARCH) && pip install -e .

apt-get update && apt-get install -y libgl1-mesa-dev libglib2.0-0 libgl1

pip install numpy==1.26.4 opencv-python==4.9.0.80

echo "Setup complete!"