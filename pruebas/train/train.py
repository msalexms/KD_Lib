import torch.optim as optim
from .distiller import get_distiller


def get_optimizer(optimizer_name, model_parameters, learning_rate, decay, momentum):
    if optimizer_name == "SGD":
        return optim.SGD(model_parameters, lr=learning_rate, weight_decay=decay, momentum=momentum)
    elif optimizer_name == "Adam":
        return optim.Adam(model_parameters, lr=learning_rate, weight_decay=decay)
    elif optimizer_name == "RMSprop":
        return optim.RMSprop(model_parameters, lr=learning_rate, weight_decay=decay, momentum=momentum)
    elif optimizer_name == "Adagrad":
        return optim.Adagrad(model_parameters, lr=learning_rate, weight_decay=decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def train_models(method_name, teacher_model, student_model, train_loader, val_loader, epochs_teacher,
                 epochs_student, learning_rate, decay, momentum, lr_decay, experiment, optimizer_name):
    teacher_optimizer = get_optimizer(optimizer_name, teacher_model.parameters(), learning_rate, decay, momentum)
    student_optimizer = get_optimizer(optimizer_name, student_model.parameters(), learning_rate, decay, momentum)

    lr_scheduler = optim.lr_scheduler.StepLR(teacher_optimizer, step_size=lr_decay, gamma=0.1)

    distiller = get_distiller(method_name, teacher_model, student_model, train_loader, val_loader, teacher_optimizer,
                              student_optimizer, lr_scheduler, device='cuda')

    distiller.train_teacher(epochs=epochs_teacher, plot_losses=False, save_model=True,
                            save_model_pth=f"../models/teacher_{experiment}.pt")
    distiller.train_student(epochs=epochs_student, plot_losses=False, save_model=True,
                            save_model_pth=f"../models/student_{experiment}.pt")
    distiller.evaluate(teacher=False)
    distiller.evaluate(teacher=True)

    return distiller
