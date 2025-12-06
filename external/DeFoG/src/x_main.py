import os
import pathlib
import warnings

import graph_tool
import torch

torch.cuda.empty_cache()
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from src import utils
from metrics.abstract_metrics import TrainAbstractMetricsDiscrete
from graph_discrete_flow_model import GraphDiscreteFlowModel
from models.extra_features import DummyExtraFeatures, ExtraFeatures


warnings.filterwarnings("ignore", category=PossibleUserWarning)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    dataset_config = cfg["dataset"]

    if dataset_config["name"] in [
        "sbm",
        "comm20",
        "planar",
        "tree",
    ]:
        from datasets.spectre_dataset import (
            SpectreGraphDataModule,
            SpectreDatasetInfos,
        )
        from analysis.spectre_utils import (
            PlanarSamplingMetrics,
            SBMSamplingMetrics,
            TreeSamplingMetrics,
        )

        datamodule = SpectreGraphDataModule(cfg)
        if dataset_config["name"] == "sbm":
            sampling_metrics = SBMSamplingMetrics(datamodule)
        elif dataset_config["name"] == "planar":
            sampling_metrics = PlanarSamplingMetrics(datamodule)
        elif dataset_config["name"] == "tree":
            sampling_metrics = TreeSamplingMetrics(datamodule)
        else:
            raise NotImplementedError(
                f"Dataset {dataset_config['name']} not implemented"
            )

        dataset_infos = SpectreDatasetInfos(datamodule, dataset_config)

        train_metrics = TrainAbstractMetricsDiscrete()

        extra_features = ExtraFeatures(
            cfg.model.extra_features,
            cfg.model.rrwp_steps,
            dataset_info=dataset_infos,
        )
        domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(
            datamodule=datamodule,
            extra_features=extra_features,
            domain_features=domain_features,
        )

    elif dataset_config["name"] in ["qm9", "guacamol", "moses", "zinc"]:
        from metrics.molecular_metrics import (
            SamplingMolecularMetrics,
        )
        from models.extra_features_molecular import ExtraMolecularFeatures

        if "qm9" in dataset_config["name"]:
            from datasets import qm9_dataset

            datamodule = qm9_dataset.QM9DataModule(cfg)
            dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)
            dataset_smiles = qm9_dataset.get_smiles(
                cfg=cfg,
                datamodule=datamodule,
                dataset_infos=dataset_infos,
                evaluate_datasets=False,
            )
        elif dataset_config["name"] == "guacamol":
            from datasets import guacamol_dataset

            datamodule = guacamol_dataset.GuacamolDataModule(cfg)
            dataset_infos = guacamol_dataset.Guacamolinfos(datamodule, cfg)
            dataset_smiles = guacamol_dataset.get_smiles(
                raw_dir=datamodule.train_dataset.raw_dir,
                filter_dataset=cfg.dataset.filter,
            )

        elif dataset_config.name == "moses":
            from datasets import moses_dataset

            datamodule = moses_dataset.MosesDataModule(cfg)
            dataset_infos = moses_dataset.MOSESinfos(datamodule, cfg)
            dataset_smiles = moses_dataset.get_smiles(
                raw_dir=datamodule.train_dataset.raw_dir,
                filter_dataset=cfg.dataset.filter,
            )
        elif "zinc" in dataset_config["name"]:
            from datasets import zinc_dataset

            datamodule = zinc_dataset.ZINCDataModule(cfg)
            dataset_infos = zinc_dataset.ZINCinfos(datamodule=datamodule, cfg=cfg)
            dataset_smiles = zinc_dataset.get_smiles(
                cfg=cfg,
                datamodule=datamodule,
                dataset_infos=dataset_infos,
                evaluate_datasets=False,
            )
        else:
            raise ValueError("Dataset not implemented")

        extra_features = ExtraFeatures(
            cfg.model.extra_features,
            cfg.model.rrwp_steps,
            dataset_info=dataset_infos,
        )
        domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)

        dataset_infos.compute_input_output_dims(
            datamodule=datamodule,
            extra_features=extra_features,
            domain_features=domain_features,
        )

        # We do not evaluate novelty during training
        add_virtual_states = "absorbing" == cfg.model.transition
        sampling_metrics = SamplingMolecularMetrics(
            dataset_infos, dataset_smiles, cfg, add_virtual_states=add_virtual_states
        )

    elif dataset_config["name"] == "tls":
        from datasets import tls_dataset
        from metrics.tls_metrics import TLSSamplingMetrics
        from analysis.visualization import NonMolecularVisualization

        datamodule = tls_dataset.TLSDataModule(cfg)
        dataset_infos = tls_dataset.TLSInfos(datamodule=datamodule)

        train_metrics = TrainAbstractMetricsDiscrete()
        extra_features = (
            ExtraFeatures(
                cfg.model.extra_features,
                cfg.model.rrwp_steps,
                dataset_info=dataset_infos,
            )
            if cfg.model.extra_features is not None
            else DummyExtraFeatures()
        )
        domain_features = DummyExtraFeatures()

        sampling_metrics = TLSSamplingMetrics(datamodule)

        visualization_tools = NonMolecularVisualization(dataset_name=cfg.dataset.name)

        dataset_infos.compute_input_output_dims(
            datamodule=datamodule,
            extra_features=extra_features,
            domain_features=domain_features,
        )

    else:
        raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))

    dataset_infos.compute_reference_metrics(
        datamodule=datamodule,
        sampling_metrics=sampling_metrics,
    )

    model_kwargs = {
        "dataset_infos": dataset_infos,
        "train_metrics": train_metrics,
        "sampling_metrics": sampling_metrics,
        "extra_features": extra_features,
        "visualization_tools": None,
        "domain_features": domain_features,
        "test_labels": (
            datamodule.test_labels
            if ("qm9" in cfg.dataset.name and cfg.general.conditional)
            else None
        ),
    }

    utils.create_folders(cfg)
    model = GraphDiscreteFlowModel(cfg=cfg, **model_kwargs)


    name = cfg.general.name
    if name == "debug":
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")

    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()
    trainer = Trainer(
        gradient_clip_val=cfg.train.clip_grad,
        strategy="ddp_find_unused_parameters_true",  # Needed to load old checkpoints
        accelerator="gpu" if use_gpu else "cpu",
        devices=cfg.general.gpus if use_gpu else 1,
        max_epochs=cfg.train.n_epochs,
    )



    # Start by evaluating test_only_path
    trainer.test(model, datamodule=datamodule, ckpt_path=cfg.general.test_only)


main()

