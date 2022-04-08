FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
ADD requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && pip install gdown
WORKDIR /pipeline
ADD Segment_mini_dev/scripts/segmentation_main.py /pipeline/segmentation_main.py
ADD Segment_mini_dev/scripts/helper_mini.py /pipeline/segmentation_main.py
RUN gdown -O /pipeline/saved_models/ https://drive.google.com/uc?id=1HBSGXbWw5Vorj82buF-gCi6S2DpF4mFL
RUN apt-get update && apt-get install wget -y
RUN wget -P /pipeline/.cache/torch/hub/checkpoints http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth
CMD python segmentation_main.py
