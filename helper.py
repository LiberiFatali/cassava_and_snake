import time
import torch
# from logger import Logger


def train_model(model, dataloaders, criterion, optimizer, device, checkpoint_save_path, num_epoch_to_stop_if_no_better,
                fold_idx=0, num_epochs=25, is_inception=False, is_resume=False):
    since = time.time()
    # logger = Logger('./logs')

    if is_resume:
        print('Loading weights from previous checkpoint...')
        checkpoint = torch.load(checkpoint_save_path + '_' + str(fold_idx))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch = checkpoint['epoch']
        # loss = checkpoint['loss']

    val_acc_history = []

    # best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    same_acc_epoch_num = 0
    last_epoch = 0
    last_loss = -1

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val':
                # # Tensorboard Logging - Log scalar values (scalar summary)
                # info = {'loss': epoch_loss, 'accuracy': epoch_acc}
                # for tag, value in info.items():
                #     logger.scalar_summary(tag, value, epoch)

                # save improved weights
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    # best_model_wts = copy.deepcopy(model.state_dict())
                    same_acc_epoch_num = 0

                    # save a checkpoint
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': epoch_loss,
                    }, checkpoint_save_path + '_' + str(fold_idx))
                else:
                    same_acc_epoch_num += 1

            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()
        last_epoch = epoch
        last_loss = epoch_loss
        if same_acc_epoch_num > num_epoch_to_stop_if_no_better:
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    # model.load_state_dict(best_model_wts)

    # save a General Checkpoint for Inference and/or Resuming Training
    torch.save({
        'epoch': last_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': last_loss,
    }, checkpoint_save_path + '_' + str(last_epoch) + '_' + str(fold_idx))

    return model, val_acc_history, best_acc
