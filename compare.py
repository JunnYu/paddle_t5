from paddlenlp.transformers.t5 import T5Model as PDT5Model
from transformers.models.t5.modeling_t5 import T5Model as PTT5Model
import torch
import paddle

paddle.set_device("cpu")

size = "large"
PREFIX = "E:/paddle论文复现/suoyoudaima/paddle_t5-old/"
pd_model = PDT5Model.from_pretrained(f"{PREFIX}paddle/t5-{size}")
pd_model.eval()
pt_model = PTT5Model.from_pretrained(f"{PREFIX}google/t5-{size}")
pt_model.eval()

with paddle.no_grad():
    pd_outputs = pd_model(
        **pd_model.dummy_inputs,return_dict=True
    ).last_hidden_state

with torch.no_grad():
    pt_outputs = pt_model(
        **pt_model.dummy_inputs
    ).last_hidden_state


def compare(a, b):
    a = torch.tensor(a.numpy()).float()
    b = torch.tensor(b.numpy()).float()
    meandif = (a - b).abs().mean()
    maxdif = (a - b).abs().max()
    print("mean difference:", meandif)
    print("max difference:", maxdif)


compare(pd_outputs, pt_outputs)
