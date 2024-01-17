import os.path
import sys
sys.path.extend([".", ".."])
from data.ATLAS_dataloader import prepare_all_data

from CONSTANTS import *
from torch import optim
from utils.processing import data_iter, generate_batched_tensor, generate_atlas_batched_tensor
from data.dataloader import prepare_data
from data.vocab import AssertionVocab
from Modules.ScheduledOptim import *
from Modules.Scheduler import PolynomialLRDecay
import wandb
from Models.ATLAS import ATLAS

logger = logging.getLogger('trainer')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler(sys.stderr)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

file_handler = logging.FileHandler(os.path.join(LOG_ROOT, 'trainer.log'))
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

logger.addHandler(console_handler)
logger.addHandler(file_handler)
logger.info(
    'Construct logger for Trainer succeeded, current working directory: %s, logs will be written in %s' %
    (os.getcwd(), LOG_ROOT))
config = Configurable(os.path.join(PROJECT_ROOT, 'config/atlas.ini'))
char_vocab = CharVocab()


def train():
    # wandb.init(project="DeepOracle_sampled")
    # wandb.init(settings=wandb.Settings(start_method="fork"))
    logger.info('Start training.')
    train_pkl = config.train_file
    if not os.path.exists(train_pkl):
        prepare_all_data()
    vocab_pkl = config.global_vocab_file
    vocab_file = open(os.path.join(PROJECT_ROOT, vocab_pkl), 'rb')
    global_vocab = pickle.load(vocab_file)
    vocab_file.close()
    # dataset, global_vocab = prepare_train_data()
    model = ATLAS(embedding_size=512, hidden_size=256, global_vocab_size=global_vocab.vocab_size,
                  assert_max_len=1000, enc_dropout=0.2, dec_dropout=0.2)
    params = [
        {"params": model.parameters(), "lr": config.learning_rate, 'notchange': False}, ]
    optimizer = optim.AdamW(params, eps=1e-8)
    scheduler = PolynomialLRDecay(optimizer, max_decay_steps=100000, end_learning_rate=0.000)
    if use_cuda:
        logger.info('using GPU')
        model = model.to(device)

    global_step = 0
    best_acc = -1
    best_loss = np.inf
    skipped_steps = set()
    tr_loss = 0.0
    avg_loss = 0.0
    for epoch in range(config.train_iters):
        logger.info('Epoch : %d' % epoch)
        epoch_start = time.time()
        train_reader = open(os.path.join(PROJECT_ROOT, train_pkl), 'rb')
        step = 0
        batch_inst = []
        eof = False
        while True:
            try:
                inst = pickle.load(train_reader)
                if inst.context_len == 0 or inst.assertion_len == 0:
                    continue
                batch_inst.append(inst)
            except:
                eof = True
            finally:
                if len(batch_inst) % config.train_batch_size == 0 or eof:
                    model.train()
                    step += 1
                    batch_start = time.time()
                    tinst = generate_atlas_batched_tensor(batch_inst, config, global_vocab)
                    if use_cuda:
                        tinst.to_cuda(device)
                    # try:
                    loss, _ = model.forward(tinst.inputs, tinst.outputs)
                    loss = torch.mean(loss)
                    loss_value = loss.item()
                    loss = loss / config.accumulation_steps
                    loss.backward()
                    # print(f"train {i} batch")
                    # except RuntimeError as rte:
                    #     batch_inst.clear()
                    #     continue
                    global_step += 1
                    batch_inst.clear()
                    # wandb.log({"average loss per step": loss_value})
                    tinst = None
                    if step % config.accumulation_steps == 0:
                        # update params
                        optimizer.step()

                        lr = optimizer.state_dict()['param_groups'][0]['lr']
                        # wandb.log({"learning_rate": lr})

                        # update lr
                        scheduler.step()

                        optimizer.zero_grad()
                        # print(f"in {i}-th batch update lr")
                    if global_step % config.validate_every == 0:
                        # print(f"in {i}-th batch valuate model")
                        logger.info('Start validating.')
                        start = time.time()
                        acc, loss = val(model, config)
                        end = time.time()
                        logger.info(
                            'Validation result: Acc: %.6f, loss: %.6f, time: %.2f' % (acc, loss, end - start))
                        # wandb.log({'validation accuracy': acc})
                        if best_loss > loss:
                            torch.save(model.state_dict(),
                                       os.path.join(outmodel_dir,
                                                    'atlas_best.ckpt'))
                            best_loss = loss
                            best_acc = acc

                    if step % config.save_every == 0:
                        torch.save(model.state_dict(),
                                   os.path.join(outmodel_dir, 'atlas_last.ckpt'))
                    batch_end = time.time()
                    if step % 1000 == 0:
                        logger.info('Step %d: Loss %.4f , time %.2f' % (step, loss_value, batch_end - batch_start))
                if eof:
                    break
        epoch_end = time.time()
        torch.save(model.state_dict(),
                   os.path.join(outmodel_dir, 'atlas_last.ckpt'))
        logger.info('Epoch %d end in %.2f.' % (epoch, epoch_end - epoch_start))
        train_reader.close()
        pass


def val(model, config):
    with torch.no_grad():
        accs = []
        losses = []
        step = 0
        dev_file = os.path.join(PROJECT_ROOT, config.dev_file)
        dev_pickle = open(dev_file, 'rb')
        vocab_pkl = config.global_vocab_file
        vocab_file = open(os.path.join(PROJECT_ROOT, vocab_pkl), 'rb')
        global_vocab = pickle.load(vocab_file)
        vocab_file.close()
        eof = False
        batch_inst = []
        while True:
            try:
                inst = pickle.load(dev_pickle)
                if inst.context_len == 0 or inst.assertion_len == 0: continue
                batch_inst.append(inst)
            except:
                eof = True
            finally:
                if len(batch_inst) % config.val_batch_size == 0 or eof:
                    batch_start = time.time()
                    tinst = generate_atlas_batched_tensor(batch_inst, config, global_vocab)
                    # if use_cuda:
                    #     tinst.to_cuda(device)
                    loss, probs = model.predict(tinst.inputs, tinst.outputs, global_vocab)
                    cur_step_loss = loss.mean().item()
                    losses.append(cur_step_loss)
                    # wandb.log({'validation loss': cur_step_loss})
                    pred = probs.argmax(dim=-1)
                    resmask = torch.gt(tinst.target_mask, 0)
                    acc = (torch.eq(pred, tinst.targets) * resmask).float()
                    resTruelen = torch.sum(resmask, dim=-1).float()
                    acc = acc.sum(dim=-1) / resTruelen
                    accs.append(acc.mean().item())
                    step += 1
                    duration = time.time() - batch_start
                    if step % 1000 == 0:
                        logger.info('Validation step %d, time %.2f' % (step, duration))
                    batch_inst.clear()
                    tinst = None
                if eof:
                    break
        acc = np.mean(accs)
        avg_loss = np.mean(losses)
        dev_pickle.close()
        return acc, avg_loss


if __name__ == "__main__":
    train()
    print("Hello world")
    pass

