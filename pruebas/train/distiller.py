from KD_Lib.KD import VanillaKD, LabelSmoothReg, ProbShift, MessyCollab, MeanTeacher, VirtualTeacher, NoisyTeacher, SoftRandom


def get_distiller(method, teacher_model, student_model, train_loader, val_loader, teacher_optimizer, student_optimizer,
                  exp_lr_scheduler, device):
    if method == "VanillaKD":
        return VanillaKD(teacher_model, student_model, train_loader, val_loader, teacher_optimizer, student_optimizer,
                         exp_lr_scheduler=exp_lr_scheduler, device=device)
    elif method == "LabelSmoothReg":
        return LabelSmoothReg(teacher_model, student_model, train_loader, val_loader, teacher_optimizer,
                              student_optimizer, device=device, exp_lr_scheduler=exp_lr_scheduler)
    elif method == "ProbShift":
        return ProbShift(teacher_model, student_model, train_loader, val_loader, teacher_optimizer, student_optimizer,
                         device=device,exp_lr_scheduler=exp_lr_scheduler)
    elif method == "MessyCollab":
        return MessyCollab(teacher_model, student_model, train_loader, val_loader, teacher_optimizer, student_optimizer, device=device)
    elif method == "MeanTeacher":
        return MeanTeacher(teacher_model, student_model, train_loader, val_loader, teacher_optimizer, student_optimizer,
                           exp_lr_scheduler=exp_lr_scheduler, device=device)
    elif method == "NoisyTeacher":
        return NoisyTeacher(teacher_model,student_model, train_loader, val_loader, teacher_optimizer, student_optimizer,
                            exp_lr_scheduler=exp_lr_scheduler, device=device)
    elif method == "VirtualTeacher":
        return VirtualTeacher(student_model, train_loader, val_loader, student_optimizer, exp_lr_scheduler, device=device)
    elif method == "SoftRandom":
        return SoftRandom(teacher_model, student_model, train_loader, val_loader, teacher_optimizer, student_optimizer,
                          exp_lr_scheduler=exp_lr_scheduler, device=device)

    else:
        raise ValueError("Unknown distillation method")
