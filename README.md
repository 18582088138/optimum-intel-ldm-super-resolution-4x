# optimum-intel-ldm-super-resolution-4x
This repo provides a simple ldm super resolution example of how to use [Optimum-intel](https://github.com/huggingface/optimum-intel) to optimize and accelerate inference of Hugging Face Model [Ldm-super-resolution](https://huggingface.co/CompVis/ldm-super-resolution-4x-openimages) with OpenVINO on Intel CPU.

## Installation 
- optimum-intel==1.5.2 (include openvino)
    - openvino
    - openvino-dev
- diffusers
- pytorch >= 1.9.1
- onnx >= 1.13.0

### Setup Environment
```
# Install optimum-intel 
conda create -n optimum-intel python=3.8
conda activate optimum-intel
python -m pip install -r requirements.txt

# Install diffusers 
# by pip
pip install --upgrade diffusers[torch]
# by conda 
conda install -c conda-forge diffusers
```

### HuggingFace Ldm super resolution pipeline
```
python HF-ldm4x-pipeline.py
```

### OpenVINO Ldm model convert to IR
```
python ov-ldm4x-model-convert.py
```

### OpenVINO Ldm super resolution pipeline
```
python ov-ldm4x-pipeline.py
```

## ToDo 
- ldm super resolution model int8 quantization by NNCF
