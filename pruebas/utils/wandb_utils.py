import wandb

def initialize_wandb(experiment, learning_rate, architecture, dataset, epochs, batch_size, optimizador, decay, dropout, momentum, lr_decay):
    wandb.init(sync_tensorboard=False,
               project='kd',
               entity='gram-uah',
               id=experiment,
               config={
                   "kd_method": "VanillaKD",
                   "tensorBoard": False,
                   "learning_rate": learning_rate,
                   "architecture": architecture,
                   "dataset": dataset,
                   "epochs": epochs,
                   "batch_size": batch_size,
                   "optim": optimizador,
                   "regularizacion l2": decay,
                   "dropout": dropout,
                   "momentum": momentum,
                   "lr_decay": lr_decay,
               }
               )

    wandb.define_metric("teacher/epoch")
    wandb.define_metric("teacher/*", step_metric="teacher/epoch")
    wandb.define_metric("student/epoch")
    wandb.define_metric("student/*", step_metric="student/epoch")

def finish_wandb():
    wandb.finish()
