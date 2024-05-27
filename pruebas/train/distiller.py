from KD_Lib.KD import VanillaKD, LabelSmoothReg, ProbShift


def get_distiller(method, teacher_model, student_model, train_loader, val_loader, teacher_optimizer, student_optimizer,
                  exp_lr_scheduler, device):
    if method == "VanillaKD":
        return VanillaKD(teacher_model, student_model, train_loader, val_loader, teacher_optimizer, student_optimizer,
                         exp_lr_scheduler=exp_lr_scheduler, device=device)
    elif method == "LabelSmoothReg":
        return LabelSmoothReg(teacher_model, student_model, train_loader, val_loader, teacher_optimizer,
                              student_optimizer, device=device)
    elif method == "ProbShift":
        return ProbShift(teacher_model, student_model, train_loader, val_loader, teacher_optimizer, student_optimizer,
                         device=device)
    else:
        raise ValueError("Unknown distillation method")
