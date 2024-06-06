import os
import torch
import pickle
import logging
import argparse
import numpy as np
import torch.nn as nn
from dataset import CustomDataset
from autoencoder import AutoEncoder, get_model


#################EXP#START#################################
import boto3
from sagemaker.session import Session
from sagemaker.experiments.run import Run, load_run

boto_session = boto3.session.Session(region_name=os.environ["AWS_REGION"])
sagemaker_session = Session(boto_session=boto_session)
#################EXP#END###################################


import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sagemaker_training import environment

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Import SMDataParallel PyTorch Modules, if applicable
backend = 'nccl'
training_env = environment.Environment()
smdataparallel_enabled = training_env.additional_framework_parameters.get('sagemaker_distributed_dataparallel_enabled', False)
if smdataparallel_enabled:
    try:
        import smdistributed.dataparallel.torch.torch_smddp
        backend = 'smddp'
        print('Using smddp as backend')
    except ImportError: 
        print('smdistributed module not available, falling back to NCCL collectives.')

class CUDANotFoundException(Exception):
    pass

class Trainer():
    
    def __init__(self, args, model, optimizer, train_loader, val_loader, scheduler, device, epoch):
        
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.device = device
        self.epoch = epoch
        
        # Loss Function
        self.criterion = nn.L1Loss().to(self.device)
        self.anomaly_calculator = nn.L1Loss(reduction="none").to(self.device)
        
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        
    def fit(self, ):
        
        self.model.to(self.device)
        best_score = 0
        
        #################EXP#START#################################
        with load_run(sagemaker_session=sagemaker_session) as run:
            run.log_parameters(vars(args))
            for epoch in range(self.epoch):
                self.model.train()
                train_loss = []
                for time, x, y in self.train_loader:
                    time, x = time.to(self.device), x.to(self.device)

                    self.optimizer.zero_grad()

                    _x = self.model(time, x)
                    t_emb, _x = self.model(time, x)
                    x = torch.cat([t_emb, x], dim=1)

                    loss = self.criterion(x, _x)
                    loss.backward()
                    self.optimizer.step()

                    train_loss.append(loss.item())

                if epoch % 10 == 0 :
                    score = self.validation(self.model, 0.95)
                    diff = self.cos(x, _x).cpu().tolist()

                    if args.local_rank == 0:
                        # logger.info(
                        #     "Processes {}/{} ({:.0f}%) of train data".format(
                        #         len(self.train_loader.sampler),
                        #         len(self.train_loader.dataset),
                        #         100.0 * len(self.train_loader.sampler) / len(self.train_loader.dataset),
                        #     )
                        # )
                        logger.info(
                            f'Epoch : [{epoch}] train_loss:{np.mean(train_loss)}, train_cos:{np.mean(diff)} val_cos:{score})'
                        )
                        print(f'Epoch : [{epoch}] train_loss:{np.mean(train_loss)}, train_cos:{np.mean(diff)} val_cos:{score})')

                        run.log_metric(
                            name="train_loss",
                            value=np.mean(train_loss),
                            step=epoch
                        )
                        run.log_metric(
                            name="train_cos",
                            value=np.mean(diff),
                            step=epoch
                        )
                        run.log_metric(
                            name="val_cos",
                            value=score,
                            step=epoch
                        )
                

                if self.scheduler is not None:
                    self.scheduler.step(score)

                if best_score < score:
                    best_score = score
                    #torch.save(model.module.state_dict(), './best_model.pth', _use_new_zipfile_serialization=False)
                    torch.save(self.model.module.state_dict(), os.path.join(self.args.model_dir, "./best_model.pth"), _use_new_zipfile_serialization=False)
                    #torch.save(self.model.state_dict(), os.path.join(self.args.model_dir, "./best_model.pth"), _use_new_zipfile_serialization=False)
                    
        #################EXP#END###################################
        
        return self.model
    
    def validation(self, eval_model, thr):
        
        eval_model.eval()
        with torch.no_grad():
            for time, x, y in self.val_loader:
                time, x, y= time.to(self.device), x.to(self.device), y.to(self.device)
                _x = self.model(time, x)
                
                t_emb, _x = self.model(time, x)
                x = torch.cat([t_emb, x], dim=1)
                
                anomal_score = self.anomaly_calculator(x, _x)
                diff = self.cos(x, _x).cpu().tolist()
                
        return np.mean(diff)

def check_gpu():
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"# DEVICE {i}: {torch.cuda.get_device_name(i)}")
            print("- Memory Usage:")
            print(f"  Allocated: {round(torch.cuda.memory_allocated(i)/1024**3,1)} GB")
            print(f"  Cached:    {round(torch.cuda.memory_reserved(i)/1024**3,1)} GB\n")

    else:
        print("# GPU is not available")

    # GPU 할당 변경하기
    #GPU_NUM = 0 # 원하는 GPU 번호 입력
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    #torch.cuda.set_device(device) # change allocation of current GPU
    #print ('# Current cuda device: ', torch.cuda.current_device()) # check
    
    #device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #torch.cuda.set_device(device) # change allocation of current GPU
    print ('# Current cuda device: ', torch.cuda.current_device()) # check
    
    return device

def from_pickle(obj_path):

    with open(file=obj_path, mode="rb") as f:
        obj=pickle.load(f)

    return obj

def get_and_define_dataset(args):
    
    train_x_scaled_shingle = from_pickle(
        obj_path=os.path.join(
            args.train_data_dir,
            "data_x_scaled_shingle.pkl"
        )
    )
    
    train_y_shingle = from_pickle(
        obj_path=os.path.join(
            args.train_data_dir,
            "data_y_shingle.pkl"
        )
    )

    train_ds = CustomDataset(
        x=train_x_scaled_shingle,
        y=train_y_shingle
    )

    test_ds = CustomDataset(
        x=train_x_scaled_shingle,
        y=train_y_shingle
    )
    
    return train_ds, test_ds

def get_dataloader(args, train_ds, test_ds):
    
    
    #################SMDDP#START#################################
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_ds,
        num_replicas=args.world_size,
        rank=rank
    )
    
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_ds,
        num_replicas=args.world_size,
        rank=rank
    )

    train_loader = torch.utils.data.DataLoader(
        dataset = train_ds,
        batch_size = args.batch_size,
        shuffle = False,
        pin_memory=True,
        num_workers=args.workers,
        prefetch_factor=3,
        sampler=train_sampler
    )

    val_loader = torch.utils.data.DataLoader(
        dataset = test_ds,
        batch_size = args.batch_size,
        shuffle = False,
        pin_memory=True,
        num_workers=args.workers,
        prefetch_factor=3,
        sampler=test_sampler
    )
    
    #################SMDDP#END###################################
    
    return train_loader, val_loader

def train(args):
    
    # if args.distributed:
    #     # Initialize the distributed environment.
    #     world_size = len(args.hosts)
    #     os.environ['WORLD_SIZE'] = str(world_size)
    #     host_rank = args.hosts.index(args.current_host)
    #     dist.init_process_group(backend=args.backend, rank=host_rank)
    
    logger.info("Check gpu..")
    device = check_gpu()
    logger.info("Device Type: {}".format(device))
    
    logger.info("Load and define dataset..")
    train_ds, test_ds = get_and_define_dataset(args)
    
    logger.info("Define dataloader..")
    train_loader, val_loader = get_dataloader(args, train_ds, test_ds)
    
    logger.info("Set components..")
    
    device = torch.device(f"cuda:{args.local_rank}")
    #model = Net().to(device)

    model = get_model(
        input_dim=args.num_features*args.shingle_size + args.emb_size,
        hidden_sizes=[64, 48],
        btl_size=32,
        emb_size=args.emb_size
    ).to(device)
    
    #################SMDDP#START#################################
    model = DDP(model, device_ids=[local_rank])
    #################SMDDP#END###################################

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=args.lr
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode='max',
        factor=0.5,
        patience=10,
        threshold_mode='abs',
        min_lr=1e-8,
        verbose=True
    )
    
    logger.info("Define trainer..")
    trainer = Trainer(
        args=args,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        device=device,
        epoch=args.epochs
    )
    
    logger.info("Start training..")
    model = trainer.fit()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--workers", type=int, default=os.environ["SM_HP_WORKERS"], metavar="W", help="number of data loading workers (default: 2)")
    parser.add_argument("--epochs", type=int, default=os.environ["SM_HP_EPOCHS"], metavar="E", help="number of total epochs to run (default: 150)")
    parser.add_argument("--batch_size", type=int, default=os.environ["SM_HP_BATCH_SIZE"], metavar="BS", help="batch size (default: 512)")
    parser.add_argument("--lr", type=float, default=os.environ["SM_HP_LR"], metavar="LR", help="initial learning rate (default: 0.001)")
    #parser.add_argument("--momentum", type=float, default=0.9, metavar="M", help="momentum (default: 0.9)")
    #parser.add_argument("--dist_backend", type=str, default="gloo", help="distributed backend (default: gloo)")
    #parser.add_argument("--hosts", type=json.loads, default=os.environ["SM_HOSTS"]) #["algo-1"]
    #parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"]) #"algo-1"
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"]) #"/opt/ml/model"
    parser.add_argument("--train_data_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"]) # /opt/ml/input/data/train
    parser.add_argument("--val_data_dir", type=str, default=os.environ["SM_CHANNEL_VALIDATION"]) # /opt/ml/input/data/valication
    parser.add_argument("--num_gpus", type=int, default=os.environ["SM_NUM_GPUS"]) #1
    
    parser.add_argument("--shingle_size", type=int, default=os.environ["SM_HP_SHINGLE_SIZE"])
    parser.add_argument("--num_features", type=int, default=os.environ["SM_HP_NUM_FEATURES"])
    parser.add_argument("--emb_size", type=int, default=os.environ["SM_HP_EMB_SIZE"])
    
    #################SMDDP#START#################################
    dist.init_process_group(backend=backend)
    args = parser.parse_args()
    args.world_size = dist.get_world_size()
    args.rank = rank = dist.get_rank()
    args.local_rank = local_rank = int(os.getenv("LOCAL_RANK", -1))
    #################SMDDP#END###################################
    print ("args.local_rank", args.local_rank)

    train(args)