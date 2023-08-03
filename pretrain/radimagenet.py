from torch.utils.data import Dataset
from typing import Union, Optional, Callable
from pathlib import Path
import pandas as pd

from torchvision.datasets.folder import default_loader


class RadImageNetSimple(Dataset):
    def __init__(
        self,
        root_dir: Union[str, Path],
        split_file: pd.DataFrame,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader_fn: Callable = default_loader,
        overfit: int = -1,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.split_file = split_file
        if overfit > 0:
            self.split_file = self.split_file.iloc[:overfit]
        self.transform = transform if transform is not None else lambda x: x
        self.target_transform = (
            target_transform if target_transform is not None else lambda x: x
        )
        self.loader_fn = loader_fn

        self.label_str_to_int = list(self.split_file.label.unique())
        self.label_str_to_int.sort()
        self.label_str_to_int = {
            label: idx for idx, label in enumerate(self.label_str_to_int)
        }

    def __len__(self) -> int:
        return len(self.split_file)

    def __getitem__(self, index: int):
        entry = self.split_file.iloc[index]
        path = self.root_dir.parent / entry.filename
        label = self.label_str_to_int[entry.label]
        image = self.loader_fn(path)
        return self.transform(image), self.target_transform(label)
