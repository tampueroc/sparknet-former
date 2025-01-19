import yaml
import argparse
import pytorch_lightning as pl
from utils import Logger, CheckpointHandler, EarlyStoppingHandler
from data import FireDataModule
from models import SparkNetFormer

def load_yaml_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main(args):
    # 1 Load all three config files
    default_cfg = load_yaml_config(args.default_config)
    model_cfg   = load_yaml_config(args.model_config)
    data_cfg    = load_yaml_config(args.data_config)

    # Configs
    trainer_cfg = default_cfg.get('trainer', {})
    early_stopper_config = trainer_cfg['early_stopper_handler']
    logger_config = trainer_cfg['logger']
    checkpoint_config = trainer_cfg['checkpoint_handler']

    # Logger
    if logger_config.get('enabled', False):
        logger = Logger.get_tensorboard_logger(
            save_dir=logger_config['dir'],
            name=logger_config['name']
        )

    # Callbacks
    checkpoint_callback = CheckpointHandler.get_checkpoint_callback(
        dirpath=checkpoint_config['dir'],
        monitor=checkpoint_config['monitor'],
        mode=checkpoint_config['mode']
    )
    early_stopping_callback = EarlyStoppingHandler.get_early_stopping_callback(
        monitor=early_stopper_config['monitor'],
        patience=early_stopper_config['patience'],
        mode=early_stopper_config['mode'],
        min_delta=early_stopper_config['min_delta']
    )

    global_params = default_cfg.get('global_params', {})
    data_params = data_cfg.get('data', {})

    # 2 Initialize the DataModule
    dm = FireDataModule(
        data_dir=data_params['data_dir'],
        sequence_length=data_params['sequence_length'],
        batch_size=data_params['batch_size'],
        num_workers=data_params['num_workers'],
        drop_last=data_params['drop_last'],
        pin_memory=data_params['pin_memory'],
        seed=global_params.get('seed', 42)
    )
    dm.setup()

    # 3 Initialize the Model
    model = SparkNetFormer(model_cfg)

    trainer = pl.Trainer(
        max_epochs=trainer_cfg['max_epochs'],
        accelerator=trainer_cfg['accelerator'],
        devices=trainer_cfg['devices'],
        precision=trainer_cfg['precision'],
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback]
    )

    trainer.fit(
            model=model,
            train_dataloaders=dm.train_dataloader(),
            val_dataloaders=dm.val_dataloader()
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--default_config", default="configs/default_config.yaml", help="Path to default config.")
    parser.add_argument("--model_config", default="configs/model_config.yaml", help="Path to model config.")
    parser.add_argument("--data_config", default="configs/data_config.yaml", help="Path to data config.")
    args = parser.parse_args()

    main(args)

