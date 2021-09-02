import os
import torch
import logging

CLASS_TO_LABEL = {
    0: "UNKNOWN",
    1: "A",
    2: "B",
    3: "C",
    4: "D",
    5: "E"
}


def save_checkpoint(model, extra, checkpoint, checkpoint_dir, optimizer=None):

    state = {'state_dict': model.state_dict(),
             'extra': extra}
    if optimizer:
        state['optimizer'] = optimizer.state_dict()

    path = os.path.join(checkpoint_dir, checkpoint)
    torch.save(state, path)
    logging.info('model saved to %s' % path)


def load_checkpoint(model, checkpoint, checkpoint_dir, device='cpu', optimizer=None):
    path = os.path.join(checkpoint_dir, checkpoint)

    exists = os.path.isfile(path)
    if exists:
        state = torch.load(path, map_location=device)

        model.load_state_dict(state['state_dict'], strict=False)
        optimizer_state = state.get('optimizer')
        if optimizer and optimizer_state:
            optimizer.load_state_dict(optimizer_state)

        logging.info("Checkpoint loaded: %s " % state['extra'])
        return state['extra']
    else:
        logging.warn("Checkpoint not found")
        return {'epoch': 0, 'lb_acc': 0}
