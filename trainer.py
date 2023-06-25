import math
import os
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union, Mapping, NewType
import logging

from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, Sampler
from torch.nn import DataParallel
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim

from .trainer_callback import (CallbackHandler,
                               TrainerState,
                               TrainerControl,
                               TrainerCallback,
                               DefaultFlowCallback,
                               LoggerCallback,
                               PrinterCallback)
from .utils import TrainingArguments, set_seed

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"

WEIGHTS_NAME = "pytorch_model.bin"
WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"
FLAX_WEIGHTS_NAME = "flax_model.msgpack"
FLAX_WEIGHTS_INDEX_NAME = "flax_model.msgpack.index.json"
SAFE_WEIGHTS_NAME = "model.safetensors"
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
CONFIG_NAME = "config.json"
FEATURE_EXTRACTOR_NAME = "preprocessor_config.json"
IMAGE_PROCESSOR_NAME = FEATURE_EXTRACTOR_NAME
MODEL_CARD_NAME = "modelcard.json"

InputDataClass = NewType("InputDataClass", Any)
DataCollator = NewType("DataCollator", Callable[[List[InputDataClass]], Dict[str, Any]])

formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(name)s- %(process)d - %(filename)s:%(funcName)s - %(message)s")

handler1 = logging.StreamHandler()
handler2 = logging.FileHandler(filename="train.log", mode="w", encoding="utf-8")

handler1.setFormatter(formatter)
handler1.setLevel(level=logging.INFO)

handler2.setFormatter(formatter)
handler2.setLevel(level=logging.INFO)

logger = logging.getLogger("train")
logger.setLevel(logging.INFO)

logger.addHandler(handler1)
logger.addHandler(handler2)


class Trainer:
    def __init__(self,
                 model: nn.Module,
                 args: TrainingArguments,
                 data_collator: DataCollator,
                 loss_fn: nn.Module = None,
                 train_dataset: Optional[Dataset] = None,
                 eval_dataset: Optional[Dataset] = None,
                 callbacks: Optional[List[TrainerCallback]] = None,
                 optimizers: torch.optim.Optimizer = None,
                 lr_scheduler: torch.optim.lr_scheduler.LambdaLR = None,
                 compute_metrics: Callable = None):
        # check the type of inputs
        if not isinstance(model, nn.Module):
            raise ValueError("The `model` should be a class of nn.Module.")

        if not isinstance(args, TrainingArguments):
            raise ValueError("The `args` should be a class of TrainingArguments.")

        if not isinstance(train_dataset, Dataset):
            raise ValueError("The `train_dataset` should be a Dataset.")

        self.args = args

        self.with_gpu = True if args.with_gpu and torch.cuda.is_available() else False
        # Seed must be set before instantiating the model when using model
        set_seed(self.args.seed)

        self.is_in_train = args.do_train

        self.data_collator = data_collator
        self.sampler = None
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset if eval_dataset is not None else train_dataset

        self.model = model

        self.optimizer = optimizers if optimizers is not None else self.get_train_optimizer()
        # self.optimizer = self.get_train_optimizer(self.model)
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn

        self.compute_metrics = compute_metrics

        if not callable(self.data_collator) and callable(getattr(self.data_collator, "collate_batch", None)):
            raise ValueError("The `data_collator` should be a simple callable (function, class with `__call__`).")

        self.state = TrainerState()
        self.control = TrainerControl()

        default_callbacks = []  # [DefaultFlowCallback]
        callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            callbacks, self.model, self.optimizer, self.lr_scheduler
        )
        # self.loggercallback = LoggerCallback("trainlogger")

        # Internal variables to keep track of the original batch size
        self.train_batch_size = args.train_batch_size
        self.eval_batch_size = args.eval_batch_size

        # set up summary
        tensorboard_dir = self.args.tensorboard_dir
        if not os.path.isdir(tensorboard_dir):
            os.makedirs(tensorboard_dir)

        self.writer = SummaryWriter(self.args.tensorboard_dir)

        self.init_DDP()  # 初始化DDP
        if self.with_gpu:
            self.model = self.wrap_model(self.model)
        else:
            self.model = self.model

    def add_callback(self, callback):
        """
        Add a callback to the current list of [`~transformer.TrainerCallback`].

        Args:
           callback (`type` or [`~transformer.TrainerCallback`]):
               A [`~transformer.TrainerCallback`] class or an instance of a [`~transformer.TrainerCallback`]. In the
               first case, will instantiate a member of that class.
        """
        self.callback_handler.add_callback(callback)

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

        return DataLoader(
            train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True if train_sampler is None else False,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory)

    def get_eval_dataloader(self) -> DataLoader:
        eval_dataset = self.eval_dataset
        data_collator = self.data_collator
        eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset, shuffle=False, drop_last=False)
        eval_sampler = eval_sampler

        return DataLoader(eval_dataset,
                          batch_size=self.eval_batch_size,
                          sampler=eval_sampler,
                          collate_fn=data_collator)

    def get_train_optimizer(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-5, weight_decay=1e-5, eps=1e-8)
        # if self.optimizer is None:
        #     optimizer = optim.AdamW(self.model.parameters(),lr=1e-5,weight_decay=1e-5,eps=1e-8)
        # else:
        #     optimizer = self.optimizer(model.parameters(),lr=self.args.learning_rate)

        return optimizer

    def init_DDP(self):
        backend = self.args.backend
        rank = self.args.local_rank
        world_size = self.args.world_size
        init_method = self.args.init_method

        if init_method.lower() == "tcp":
            init_method = f"tcp://{self.args.MASTER_ADDR}:{self.args.MASTER_PORT}"
        elif init_method.lower() == "file":
            init_method = 'file://path'
        elif init_method.lower() == "env":
            os.environ["MASTER_ADDR"] = self.args.MASTER_ADDR
            os.environ["MASTER_PORT"] = self.args.MASTER_PORT
            init_method = "env://"
        else:
            raise ValueError("init_method must only be tcp,file or env.")

        dist.init_process_group(backend=backend, init_method=init_method, rank=rank, world_size=world_size)

    def wrap_model(self, model):
        rank = dist.get_rank()
        model = model.to(rank)
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
        return model

    def train(
            self,
            checkpoint: str = None):
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (`str` or `bool`, *optional*):
                If a `str`, local path to a saved checkpoint as saved by a previous instance of [`Trainer`].
        """
        model = self.model
        train_dataloader = self.get_train_dataloader()
        optimizer = self.optimizer  # self.get_train_optimizer(model)

        # 从断点加载
        if checkpoint:
            self.model.load_state_dict(os.path.join(checkpoint, WEIGHTS_NAME))
            self._load_optimizer_and_scheduler(checkpoint, optimizer, self.lr_scheduler)
            self.state = TrainerState.load_from_json(os.path.join(checkpoint, TRAINER_STATE_NAME))

        num_examples = len(self.train_dataset)
        num_train_epochs = self.args.epochs
        total_train_batch_size = self.train_batch_size * self.args.world_size * self.args.gradient_accumulation_steps
        max_steps = max(num_train_epochs * num_examples // total_train_batch_size, 1)
        num_update_steps_per_epoch = num_examples // total_train_batch_size if self.args.dataloader_drop_last else math.ceil(
            num_examples / total_train_batch_size)

        # Train!
        self.logging("***** Running training *****")
        self.logging(f"  Num examples = {num_examples}")
        self.logging(f"  Num Epochs = {num_train_epochs}")
        self.logging(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        self.logging(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        self.logging(f"  Total optimization steps = {max_steps}")
        self.logging(
            f"  Number of trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        # self.state.epoch = 0

        epochs_trained = self.state.global_step // total_train_batch_size  # 后期用于从checkpoint继续训练
        global_step = self.state.global_step

        steps_trained_in_current_epoch = global_step % num_update_steps_per_epoch

        self.control = self.callback_handler.on_train_begin(self.args, self.state, self.control)

        # tr_loss = torch.Tensor(0.0)
        for epoch in range(epochs_trained, num_train_epochs):
            # if dist.get_rank() == 0:
            #     steps_trained_progress_bar = tqdm(total=num_examples//total_train_batch_size,
            #                                       desc=f"Traing the model, in epoch of {epoch}",
            #                                       colour="#0396ff",
            #                                       ncols=120)
            train_dataloader.sampler.set_epoch(epoch)

            model.zero_grad()
            self.control = self.callback_handler.on_epoch_begin(self.args, self.state, self.control)
            tr_loss = 0.0
            for step, inputs in enumerate(train_dataloader):
                # 跳过已训练的
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                if step % self.args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(self.args, self.state, self.control)

                tr_loss_step = self.training_step(model, inputs)

                if dist.is_initialized() and dist.get_world_size() > 1:
                    self.all_reduce(tr_loss_step, dist.ReduceOp.SUM)
                    # dist.all_reduce(tr_loss_step, op=dist.ReduceOp.SUM)
                    tr_loss_step = tr_loss_step / self.args.world_size

                tr_loss += tr_loss_step.item()

                # 梯度累加
                if step % self.args.gradient_accumulation_steps == 0 or (
                        step <= self.args.gradient_accumulation_steps and step + 1 == num_update_steps_per_epoch
                ):
                    optimizer.step()
                    optimizer.zero_grad()

                    self.tb_add_scalar(tag="loss", scalar_value=tr_loss)

                    # if dist.get_rank() == 0:
                    #     steps_trained_progress_bar.set_postfix({"loss": "{:.5f}".format(tr_loss.item()),
                    #                                             "lr": "{:.1e}".format(optimizer.state_dict()["param_groups"][0]['lr'])})
                    #     steps_trained_progress_bar.update(1)

                    if global_step != 0 and global_step % self.args.print_log_per_steps == 0:
                        message = "epoch: {:<5},global_step: {:<5},learning_rate: {:<5.1e},loss: {:<10.5f}". \
                            format(epoch, global_step, self.get_learning_rate()["lr0"], tr_loss)
                        self.logging(message)

                    tr_loss = 0.0
                    global_step += 1
                    self.control = self.callback_handler.on_step_end(self.args, self.state, self.control)

                    self.state.global_step = global_step

                # evaluating model
                if global_step % self.args.save_peer_step == 0 and self.args.do_eval:
                    self.logging("***** Running evaluating *****.")
                    metric, evaluate_loss = self.evaluate()

                    if metric is None:
                        self.logging("eval_loss:{:.5f}".format(evaluate_loss))
                        if evaluate_loss < self.state.best_metric:
                            self.save_checkpoint(path="./checkpoint/best")
                    else:
                        message = ""
                        for k, v in metric.items():
                            message += "{}:{:<5.5f}".format(k, v)
                        self.logging(message + ",eval_loss:{:.5f}".format(evaluate_loss))
                        if metric["f1"] < self.state.best_metric:
                            self.save_checkpoint(path="./checkpoint/best")

                dist.barrier()  # 同步每个step

            # 调整学习率
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # if dist.get_rank() == 0:
            #     steps_trained_progress_bar.close()

            self.control = self.callback_handler.on_epoch_end(self.args, self.state, self.control)
            dist.barrier()  # 同步每个epoch

        self.control = self.callback_handler.on_train_end(self.args, self.state, self.control)
        self.save_checkpoint()

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        loss = self.compute_loss(model, inputs)

        loss = loss / self.args.gradient_accumulation_steps
        loss.backward()

        return loss.detach()  # 切断反向传播路径

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """

        if "labels" in inputs.keys():
            labels = inputs.pop("labels")
        else:
            labels = None  # raise ValueError("The inputs did not contain the key of labels.")

        outputs = model(**inputs)

        loss = self.loss_fn(outputs, labels)

        return (loss, outputs, labels) if return_outputs else loss

    def _prepare_input(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            if self.with_gpu:
                data = data.to(self.args.local_rank)
            return data
        return data

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError(
                "The batch received was empty, your model won't be able to train on it. Double-check that your "
                f"training dataset contains keys expected by the model: {','.join(self._signature_columns)}."
            )
        return inputs

    def save_checkpoint(self, path=None):
        if dist.is_initialized() and dist.get_rank() == 0:  # 只允许主进程保存
            if path is None:
                output_dir = os.path.join(self.args.checkpoint_dir, f"model-{self.state.global_step}")
            else:
                output_dir = path

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            if hasattr(self.model, "module"):
                torch.save(self.model.module.state_dict(), os.path.join(output_dir, WEIGHTS_NAME))
            else:
                torch.save(self.model.state_dict(), os.path.join(output_dir, WEIGHTS_NAME))

            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
            if self.lr_scheduler is not None:
                torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))

    def _load_optimizer_and_scheduler(self, checkpoint, optimizer, lr_scheduler=None):
        optimizer.load_state_dict(torch.load(os.path.join(checkpoint, OPTIMIZER_NAME)))

        if lr_scheduler:
            self.lr_scheduler.load_state_dict(torch.load(os.path.join(checkpoint, SCHEDULER_NAME)))

    def evaluate(self) -> Dict[str, float]:
        model = self.model

        evaldata_length = len(self.eval_dataset)
        total_eval_batch_size = self.train_batch_size * self.args.world_size
        eval_dataloader = self.get_eval_dataloader()
        total_steps = math.ceil(evaldata_length // total_eval_batch_size)

        if dist.get_rank() == 0:
            self.logging("Evaluating the model.")

        predict_host = list()
        label_host = list()

        tr_loss = 0.0
        for step, inputs in enumerate(eval_dataloader):
            model.eval()
            inputs = self._prepare_inputs(inputs)

            loss, predicts, labels = self.compute_loss(model, inputs, return_outputs=True)

            self.all_reduce(loss, dist.ReduceOp.SUM)
            tr_loss += loss.item() / dist.get_world_size()

            all_predicts = self.all_gather(predicts)
            all_predicts = self.tensor2numpy(all_predicts)

            all_labels = self.all_gather(labels)
            all_labels = self.tensor2numpy(all_labels)

            predict_host.append(all_predicts)
            label_host.append(all_labels)

        metrics = self.compute_metrics(predicts=predict_host, labels=label_host)
        tr_loss = tr_loss / total_steps

        return metrics, tr_loss

    def all_reduce(self, tensor: torch.Tensor, reduceop: dist.ReduceOp = None):
        dist.all_reduce(tensor, op=reduceop)

    def all_gather(self, tensor: Any):
        if isinstance(tensor, (tuple, list)):
            return type(tensor)(self.all_gather(t) for t in tensor)
        else:
            output_tensor = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(output_tensor, tensor)
            output_tensor = torch.cat(output_tensor, dim=0)
        return output_tensor

    def tensor2numpy(self, tensor: Any):
        if isinstance(tensor, (tuple, list)):
            return type(tensor)(self.tensor2numpy(t) for t in tensor)
        else:
            return tensor.data.cpu().numpy()

    # def cleanup(self):
    #     dist.barrier()
    #     dist.destroy_process_group()

    def tb_add_scalar(self, tag, scalar_value):
        self.writer.add_scalar(tag=tag, scalar_value=scalar_value, global_step=self.state.global_step)

    def logging(self, message, level="info"):
        if dist.get_rank() == 0:
            if hasattr(logger, level):
                getattr(logger, level)(message)
            else:
                logger.warning(f"logger don't have attr {level},we will output the info of logger.")
                logger.info(message)

    def get_learning_rate(self):
        lr = dict()
        param_groups = self.optimizer.state_dict()['param_groups']  # [0]['lr']
        for i, p in enumerate(param_groups):
            lr["lr" + str(i)] = p["lr"]
        return lr
