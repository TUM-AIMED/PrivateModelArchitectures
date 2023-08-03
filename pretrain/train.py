import sys
import torch
import poutyne
from numpy import argmax
from torchvision import transforms
from pathlib import Path
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from pandas import read_csv


sys.path.insert(0, str(Path.cwd()))
from PrivateModelArchitectures.classification import ResNet9
from pretrain.radimagenet import RadImageNetSimple
from datetime import datetime

train_bs, val_bs, test_bs = 64, 64, 64
default_loader_kwargs = {
    "pin_memory": True,
    "num_workers": 16,
    "prefetch_factor": 8,
}
epochs = 100
overfit = -1
id_str = f"training_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

model = ResNet9(3, 165, scale_norm=True, norm_layer="group")
optimizer = torch.optim.NAdam(model.parameters(), lr=2e-3)
loss_fn = torch.nn.CrossEntropyLoss()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


root_path = Path("./pretrain/data/radiology_ai/")
train_ds, val_ds, test_ds = [
    RadImageNetSimple(
        root_dir=root_path,
        split_file=read_csv(root_path / f"RadiologyAI_{split}.csv"),
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(0.22039941, 0.24865805)]
        ),
        overfit=overfit,
    )
    for split in ["train", "val", "test"]
]
# train_dl = torch.utils.data.DataLoader(
#     train_ds, batch_size=train_bs, shuffle=True, **default_loader_kwargs
# )
# val_dl, test_dl = [
#     torch.utils.data.DataLoader(
#         dl, batch_size=bs, shuffle=False, **default_loader_kwargs
#     )
#     for dl, bs in zip((val_ds, test_ds), (val_bs, test_bs))
# ]

base_save_path = Path(f"./model_checkpoints/{id_str}")
base_save_path.mkdir(exist_ok=True, parents=True)

general_callbacks = [
    poutyne.ProgressionCallback(),
    poutyne.WandBLogger(project="PretrainingPrivateArchitectures"),
    poutyne.CSVLogger(str(base_save_path / "train_log.csv")),
]

train_callbacks = [
    poutyne.ModelCheckpoint(str(base_save_path / "last_epoch.ckpt")),
    poutyne.ModelCheckpoint(
        str(base_save_path / "best_epoch_{epoch}.ckpt"),
        monitor="val_mcc",
        mode="max",
        save_best_only=True,
        restore_best=True,
        verbose=True,
    ),
    poutyne.EarlyStopping(min_delta=0.0, patience=5),
    poutyne.ReduceLROnPlateau(),
]


def calc_mcc_poutyne(y_true, y_pred, **kwargs):
    y_pred = argmax(y_pred, axis=1)
    return matthews_corrcoef(y_true, y_pred, **kwargs)


# def calc_rocauc_poutyne(y_true, y_pred, **kwargs):
#     y_pred = torch.softmax(torch.from_numpy(y_pred), dim=1).numpy()
#     return roc_auc_score(
#         y_true, y_pred, multi_class="ovr", labels=list(range(165)), **kwargs
#     )


sklearn_metrics = poutyne.SKLearnMetrics(funcs=[calc_mcc_poutyne], names=["mcc"])


poutyne_model = poutyne.Model(
    model,
    optimizer,
    loss_fn,
    batch_metrics=["accuracy", sklearn_metrics],
    epoch_metrics=["accuracy", "top1", "f1"],
    device=device,
)
poutyne_model.fit_dataset(
    train_ds,
    val_ds,
    epochs=epochs,
    batch_size=train_bs,
    dataloader_kwargs=default_loader_kwargs,
    callbacks=train_callbacks + general_callbacks,
)


result = poutyne_model.evaluate_dataset(
    test_ds,
    batch_size=test_bs,
    dataloader_kwargs=default_loader_kwargs,
    callbacks=general_callbacks,
    return_dict_format=True,
)
print(result)
torch.save(model.state_dict(), base_save_path / "final_weights.pt")
