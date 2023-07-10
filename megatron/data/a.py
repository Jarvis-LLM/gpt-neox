from indexed_dataset import make_dataset as make_indexed_dataset

data_prefix = "/fs/archive/share/yulan/tokenized_data/yulan_v1/pilecc/train_0_text_document"
data_impl = "mmap"
skip_warmup = True
from IPython import embed
indexed_dataset = make_indexed_dataset(data_prefix, data_impl, skip_warmup)
embed()
input()