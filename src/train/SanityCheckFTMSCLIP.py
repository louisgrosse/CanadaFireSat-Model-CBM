from ssl4eos12_dataset import SSL4EOS12Dataset, collate_fn, S2L1C_MEAN, S2L1C_STD
from torchvision import transforms
from torch.utils.data import DataLoader

tf = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.Normalize(S2L1C_MEAN, S2L1C_STD),
])

ds = SSL4EOS12Dataset(
    data_dir="data/ssl4eo-s12/train",
    split_file="data/ssl4eo-s12/splits/ssl4eos12_train.txt",
    modalities=["S2L1C","captions"],
    transform=tf,
    single_timestamp=True,
)

dl = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate_fn)
batch = next(iter(dl))
print(batch.keys())           # dict_keys(['S2L1C','captions'])
print(batch["S2L1C"].shape)   # torch.Size([128, C, 224, 224]) since 2×(~64 samples/chunk)
print(type(batch["captions"]))# numpy array/list of strings of length ~128
