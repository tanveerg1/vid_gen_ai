This uses local gemma3:4b setup using ollama to generate viral tiktok, and instagram videos

## Conda commands
Example
1. Create conda env # py311
`conda create -n gemma-env python=3.12`
2. activate conda env
`conda activate gemma-env`
## To Use this
1. First Run
`pip install requirements.txt`

2. RUN
`streamlit run interface.py`


## Issues

### Fix 1
`pip install "transformers==4.44.2" "diffusers>=0.29.0" "huggingface_hub<0.26.0"`
