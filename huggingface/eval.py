### Push le model sur github
### Une fois en local 
from lm_evaluation_harness import main 
from transformers import Trainer
path = "/save/model/checkpoint.jsp"
trainer = Trainer()
model = Trainer.from_pretrained(path) 
model.push_to_hub(commit_message="model trained") 

### Puis sur CMD lancer ca 

cd lm-evaluation-harness
pip install -e .

# pour se login
huggingface-cli login

# test sur hellaswag task: finit la phrase
python main.py \
    --model nom_du_model (hf-causal) \
    --model_args pretrained=EleutherAI/gpt-j-6B \
    --tasks hellaswag \
    --device cuda:0