import torch

from options  import stage1_opts
from utils    import logger, recorders
from datasets import custom_data_loader
from models   import custom_model, solver_utils, model_utils

import train_stage1 as train_utils
import test_stage1  as test_utils
import math
args = stage1_opts.TrainOpts().parse()
log  = logger.Logger(args)


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    if hasattr(args, 'warmup') and epoch < args.warmup:
        lr = lr / (args.warmup - epoch)
    elif not args.disable_cos:
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup) / (args.epochs - args.warmup)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main(args):
    model = custom_model.buildModel(args)
    #print('XXXXXXXModel', model)
    optimizer, scheduler, records = solver_utils.configOptimizer(args, model)
    criterion = solver_utils.Stage1ClsCrit(args)
    recorder  = recorders.Records(args.log_dir, records)

    train_loader, val_loader = custom_data_loader.customDataloader(args)

    #if (not args.no_cuda) and torch.cuda.is_available():
    #    torch.cuda.set_device(args.gpu_id)
    #    model.cuda(args.gpu_id)
    #    #criterion = criterion.cuda(args.gpu_id)

    for epoch in range(args.start_epoch, args.epochs+1):
        scheduler.step()
        recorder.insertRecord('train', 'lr', epoch, scheduler.get_lr()[0])
        adjust_learning_rate(optimizer, epoch, args)
        train_utils.train(args, train_loader, model, criterion, optimizer, log, epoch, recorder)
        if epoch % args.save_intv == 0: 
            model_utils.saveCheckpoint(args.cp_dir, epoch, model, optimizer, recorder.records, args)
        log.plotCurves(recorder, 'train')

        if epoch % args.val_intv == 0:
            test_utils.test(args, 'val', val_loader, model, log, epoch, recorder)
            log.plotCurves(recorder, 'val')

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    main(args)
