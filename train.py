import os
import torch
import pytorch_lightning as pl
#from pytorch_lightning.strategies.ddp_spawn import DDPSpawnStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import argparse
import yaml
import datetime
from shutil import copyfile

from se3set.model import Lit_SE3Set
from se3set.utils import save_config


def train(config, config_dataset, load_ckpt_path, suffix_continue_path, **kwargs):
    with open(config, 'r') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    dataset_name: str
    if config_dataset is not None and os.path.exists(config_dataset):
        with open(config_dataset, 'r') as f:
            configs_dataset = yaml.load(f, Loader=yaml.FullLoader)
        dataset_name = configs_dataset["dataset_name"]
    elif os.path.exists(os.path.join(configs['train_params']['save_path'], 'input_dataset.yaml')):
        with open(os.path.join(configs['train_params']['save_path'], 'input_dataset.yaml'), 'r') as f:
            configs_dataset = yaml.load(f, Loader=yaml.FullLoader)
        dataset_name = configs_dataset["dataset_name"]
    else:
        raise ValueError("No dataset configs, use default parameters in model/layers/config.py and dataset_name in config file")
    
    # Load dataset
    if 'QM9' in dataset_name:
        from se3set.data.qm9 import LightningQM9Dataset as dataset_class
    elif 'MD17' in dataset_name:
        from se3set.data.md17 import LightningMD17Dataset as dataset_class
    elif 'MD22' in dataset_name:
        from se3set.data.md22 import LightningMD22Dataset as dataset_class
    else:
        raise NotImplementedError('Invalid dataset name: {}'.format(configs['dataset_name']))
    dataset = dataset_class(**configs['data_params'])

    # If job failed, reload the previous ckpt
    if configs['train_params'].get('load_ckpt', None) is None and load_ckpt_path is None and os.path.exists(configs['train_params']['save_path']):
        epoch_max = 0
        ckpt_file = None
        for file_ in os.listdir(configs['train_params']['save_path']):
            if file_.endswith('.ckpt'):
                cur = int(file_.split('-')[0].split('epoch=')[1])
                epoch_max = cur if cur >= epoch_max else epoch_max
                ckpt_file = file_

        if ckpt_file is not None:
            configs['train_params']['load_ckpt'] = os.path.join(configs['train_params']['save_path'], ckpt_file)

        if suffix_continue_path is not None:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            configs['train_params']['save_path'] = configs['train_params']['save_path'] + "_{}_{}".format(suffix_continue_path, timestamp)
            os.makedirs(os.path.join(configs['train_params']['save_path'], "load_ckpt"), exist_ok=True)
            ckpt_name = os.path.basename(configs['train_params']['load_ckpt'])
            copyfile(configs['train_params']['load_ckpt'], os.path.join(configs['train_params']['save_path'], "load_ckpt", ckpt_name))
            save_config(configs, os.path.join(configs['train_params']['save_path'], 'input.yaml'))
            save_config(configs_dataset, os.path.join(configs['train_params']['save_path'], 'input_dataset.yaml'))

    # Train
    pl.seed_everything(configs['seed'])
    
    task_mean, task_std = None, None
    if hasattr(dataset, 'task_mean') and hasattr(dataset, 'task_std'):
        task_mean, task_std = dataset.task_mean, dataset.task_std

    model_params = configs['model_params']

    if load_ckpt_path is not None:
        if not os.path.exists(load_ckpt_path):
            raise FileNotFoundError('Invalid checkpoint path: {}'.format(load_ckpt_path))
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        configs['train_params']['save_path'] = os.path.dirname(load_ckpt_path) + "_continue_{}".format(timestamp) if configs['mode'] == 'train' else os.path.dirname(load_ckpt_path) + "_test_{}".format(timestamp)
        os.makedirs(os.path.join(configs['train_params']['save_path'], "load_ckpt"), exist_ok=True)
        ckpt_name = os.path.basename(load_ckpt_path)
        copyfile(load_ckpt_path, os.path.join(configs['train_params']['save_path'], "load_ckpt", ckpt_name))

        save_config(configs, os.path.join(configs['train_params']['save_path'], 'input.yaml'))
        save_config(configs_dataset, os.path.join(configs['train_params']['save_path'], 'input_dataset.yaml'))

        model = Lit_SE3Set.load_from_checkpoint(load_ckpt_path, 
                                                train_params=configs['train_params'],
                                                dataset_statistics=configs_dataset, 
                                                task_mean=task_mean, 
                                                task_std=task_std)
    else:
        # save train configs
        os.makedirs(configs['train_params']['save_path'], exist_ok=True)
        save_config(configs, os.path.join(configs['train_params']['save_path'], 'input.yaml'))
        save_config(configs_dataset, os.path.join(configs['train_params']['save_path'], 'input_dataset.yaml'))
        
        model = Lit_SE3Set(model_params=model_params, 
                           train_params=configs['train_params'], 
                           dataset_statistics=configs_dataset, 
                           task_mean=task_mean, 
                           task_std=task_std)
        
    assert model is not None, 'Failed to initialize the model'

    trainer = pl.Trainer(accelerator=configs['train_params']['accelerator'], 
                         devices=configs['train_params']['num_devices'], 
                         strategy="ddp_spawn" if configs['train_params']['ddp'] else "auto", 
                         # auto_scale_batch_size=configs['train_params'].get('auto_scale_batch_size', None), 
                         # auto_lr_find=False, 
                         max_epochs=configs['train_params']['epochs'], 
                         log_every_n_steps=configs['train_params'].get('log_every_n_steps', 50), 
                         callbacks=[ModelCheckpoint(monitor='val_loss', 
                                                    save_top_k=5, 
                                                    dirpath=configs['train_params']['save_path'], 
                                                    filename='{epoch}-{val_loss:.7f}'), 
                                    #EarlyStopping(monitor='val_loss', 
                                    #              patience=configs['train_params']['early_stopping_patience'], 
                                    #              verbose=True), 
                                    LearningRateMonitor(logging_interval='epoch', log_momentum=False)], 
                         max_time=getattr(configs['train_params'], 'max_time', None),
                         # profiler='pytorch', 
                         enable_progress_bar=True, 
                         inference_mode=True if "QM9" in dataset_name else False, 
                         logger=[#TensorBoardLogger(save_dir=os.path.join(configs['train_params']['save_path'], 'tensorboard'), name=''), 
                                 CSVLogger(save_dir=configs['train_params']['save_path'], name='')])
                         #resume_from_checkpoint=configs['train_params'].get('load_ckpt', None),)

    # trainer.tune(model)

    if configs['mode'] == 'train':
        trainer.fit(model, datamodule=dataset, ckpt_path=configs['train_params'].get('load_ckpt', None))
        trainer.test(ckpt_path='best', datamodule=dataset)
    elif configs['mode'] == 'test':
        trainer.test(model, ckpt_path=configs['train_params']['load_ckpt'], datamodule=dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SE3Set is a hypergraph neural network for molecular representation.')
    parser.add_argument('--conf', '-c', type=str, help='Configuration yaml file')
    parser.add_argument('--conf_dataset', '-cd', type=str, help='Dataset configuration yaml file', default=None)
    parser.add_argument('--load_ckpt_path', '-lc', type=str, help='Load a specified checkpoint to continue the training process', default=None)
    parser.add_argument('--suffix_continue_path', '-s', type=str, help='Save the continue training results in new folder with suffix', default=None)

    args = parser.parse_args()

    train(args.conf, args.conf_dataset, args.load_ckpt_path, args.suffix_continue_path)
