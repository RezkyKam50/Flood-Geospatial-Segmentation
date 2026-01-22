import torch
import terratorch
import albumentations
import lightning.pytorch as pl
import torch.nn as nn
import torch.nn.functional as F 
from torch.nn.modules.loss import _Loss
import torch.linalg as LA
import numpy as np
from pathlib import Path
from terratorch.registry import BACKBONE_REGISTRY, TERRATORCH_BACKBONE_REGISTRY, TERRATORCH_DECODER_REGISTRY
from terratorch.datamodules import Sen1Floods11NonGeoDataModule, GenericNonGeoSegmentationDataModule
from terratorch.datasets import Sen1Floods11NonGeo
from terratorch.tasks import SemanticSegmentationTask

def load_data_module():
    dataset_path = Path('datasets/v1.1')

    datamodule = GenericNonGeoSegmentationDataModule(
        batch_size=16,
        num_workers=16,
        num_classes=2,    

        # Define data paths
        train_data_root=dataset_path / 'data/S2L1CHand',
        train_label_data_root=dataset_path / 'data/LabelHand',
        val_data_root=dataset_path / 'data/S2L1CHand',
        val_label_data_root=dataset_path / 'data/LabelHand',
        test_data_root=dataset_path / 'data/S2L1CHand',
        test_label_data_root=dataset_path / 'data/LabelHand',

        # Define splits as all samples are saved in the same folder
        train_split=dataset_path / 'splits/flood_train_data.txt',
        val_split=dataset_path / 'splits/flood_valid_data.txt',
        test_split=dataset_path / 'splits/flood_test_data.txt',
        
        # Define suffix
        img_grep='*_S2Hand.tif',
        label_grep='*_LabelHand.tif',
        
        train_transform=[
            albumentations.HorizontalFlip(),
            albumentations.Resize(224, 224),
            albumentations.pytorch.transforms.ToTensorV2(),
        ],
        val_transform=None,  # Using ToTensor() by default
        test_transform=None,
        
        # Define bands in the data and which one you want to use (optional)
        dataset_bands=[
        "COASTAL_AEROSOL",
        "BLUE",
        "GREEN",
        "RED",
        "RED_EDGE_1",
        "RED_EDGE_2",
        "RED_EDGE_3",
        "NIR_BROAD",
        "NIR_NARROW",
        "CIRRUS",
        "SWIR_1",
        "SWIR_2",
        ],
        output_bands=[
        "BLUE",
        "GREEN",
        "RED",
        "NIR_NARROW",
        "SWIR_1",
        "SWIR_2", 
        ],
        
        # Define standardization values for the output_bands
        means=[
        0.11076498225107874,
        0.13456047562676646,
        0.12477149645635542,
        0.3248933937526503,
        0.23118412840904512,
        0.15624583324071273,
        ],
        stds=[
        0.15469174852002912,
        0.13070592427323752,
        0.12786689586224442,
        0.13925781946803198,
        0.11303782829438778,
        0.10207461132314981,
        ],
    )

    return datamodule

def setup():
    datamodule = load_data_module()
    # Setup train and val datasets
    datamodule.setup("fit")

    # checking datasets train split size
    train_dataset = datamodule.train_dataset
    print(len(train_dataset))

    # checking datasets validation split size
    val_dataset = datamodule.val_dataset
    print(len(val_dataset))
    # plotting a few samples
    val_dataset.plot(val_dataset[0])
    val_dataset.plot(val_dataset[9])
    val_dataset.plot(val_dataset[11])

    # checking datasets testing split size
    datamodule.setup("test")
    test_dataset = datamodule.test_dataset
    print(len(test_dataset))


    print(list(TERRATORCH_BACKBONE_REGISTRY)[:5])

    print(list(TERRATORCH_DECODER_REGISTRY))

    # Build PyTorch model for custom pipeline
    model = BACKBONE_REGISTRY.build("prithvi_eo_v2_300_tl", pretrained=True)

    print(model)

    pl.seed_everything(0)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="output/sen1floods11/checkpoints/",
        mode="max",
        monitor="val/Multiclass_Jaccard_Index", # Variable to monitor
        filename="best-{epoch:02d}",
    )

    # Lightning Trainer
    trainer = pl.Trainer(
        accelerator="cpu",
        strategy="auto",
        devices=1, # Deactivate multi-gpu because it often fails in notebooks
        precision='bf16',  # Speed up training
        num_nodes=1,
        logger=True,  # Uses TensorBoard by default
        max_epochs=20, # For demos
        log_every_n_steps=1,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback, pl.callbacks.RichProgressBar()],
        default_root_dir="output/sen1floods11/",
    )
    return trainer, datamodule, test_dataset

def train(model, indicies):
    trainer, datamodule, test_dataset = setup()
    # Model
    model = SemanticSegmentationTask(
        model_factory="EncoderDecoderFactory",
        model_args={
            # Backbone
            "backbone": model, # Model can be either prithvi_eo_v1_100, prithvi_eo_v2_300, prithvi_eo_v2_300_tl, prithvi_eo_v2_600, prithvi_eo_v2_600_tl
            "backbone_pretrained": True,
            "backbone_num_frames": 1, # 1 is the default value
            "backbone_img_size": 512, # if not provided: interpolate pos embedding from 224 pre-training which also works well
            "backbone_bands": ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"],
            "backbone_coords_encoding": [], # use ["time", "location"] for time and location metadata
            
            # Necks 
            "necks": [
                {
                    "name": "SelectIndices",
                    "indices": indicies  
                },
                {"name": "ReshapeTokensToImage"}     
            ],
            
            # Decoder
            "decoder": "UNetDecoder",
            "decoder_channels": [512, 256, 128, 64],
            
            # Head
            "head_dropout": 0.1,
            "num_classes": 2,
        },

        loss="dice",
        optimizer="AdamW",
        ignore_index=-1,
        lr=1e-4,
        scheduler="StepLR",
        scheduler_hparams={"step_size": 10, "gamma": 0.9},
        freeze_backbone=True, 
        freeze_decoder=False,
        plot_on_val=False,
        class_names=['no water', 'water']   
    )
    trainer.fit(model, datamodule=datamodule)

    best_ckpt_path = "output/sen1floods11/checkpoints/best-epoch=01.ckpt"

    trainer.test(model, datamodule=datamodule, ckpt_path=best_ckpt_path)
    
    model = SemanticSegmentationTask.load_from_checkpoint(
        best_ckpt_path,
        model_factory=model.hparams.model_factory,
        model_args=model.hparams.model_args,
    )

    test_loader = datamodule.test_dataloader()
    with torch.no_grad():
        batch = next(iter(test_loader))
        images = batch["image"].to(model.device)
        masks = batch["mask"].numpy()

        outputs = model(images)
        preds = torch.argmax(outputs.output, dim=1).cpu().numpy()

    for i in range(5):
        sample = {key: batch[key][i] for key in batch}
        sample["prediction"] = preds[i]
        test_dataset.plot(sample)


if __name__ == "__main__":

    # "indices": [2, 5, 8, 11] # indices for prithvi_eo_v1_100
    # "indices": [5, 11, 17, 23] # indices for prithvi_eo_v2_300 or prithvi_eo_v2_600
    # "indices": [7, 15, 23, 31] # indices for prithvi_eo_v2_600

    indicies = [5, 11, 17, 23]
    model = "prithvi_eo_v2_300_tl"

    train(model, indicies)


