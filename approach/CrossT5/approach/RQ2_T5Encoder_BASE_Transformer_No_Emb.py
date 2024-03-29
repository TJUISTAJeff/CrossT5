import os.path
import sys
import traceback

sys.path.extend([".", ".."])
from torch import optim
from Models.Pretrained_Generator import DeepOracle_With_T5_Base
from Modules.BeamSearch import BeamSearch
import wandb
import time
import random
import pickle
import numpy as np
from utils.Config import Configurable
from accelerate import Accelerator
from Modules.SumDataset import SumDataset, readpickle, rs_collate_fn_t5
import torch

print('Seeding everything...')
seed = 666
random.seed(seed)  # Python random module.
np.random.seed(seed)  # Numpy module.
torch.manual_seed(seed)  # Torch CPU random seed module.
torch.cuda.manual_seed(seed)  # Torch GPU random seed module.
torch.cuda.manual_seed_all(seed)  # Torch multi-GPU random seed module.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['PYTHONHASHSEED'] = str(seed)

print('Seeding Finished')
# Device configuration
use_cuda = False
if torch.cuda.is_available():
    use_cuda = True
accelerator = Accelerator()

config = Configurable('config/t5encoder_base_transformer_no_emb.ini')
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5-base')
from tqdm import tqdm

# Model parameters
outmodel_dir = config.save_dir

# CharVocab()

device = None


def train():
    global device
    if accelerator.is_main_process:
        wandb.init(entity='ista_oracle_team', project="DeepOracle", name='t5_base_transformer_decoder')
    device = accelerator.device
    print('Start training.')
    model = DeepOracle_With_T5_Base(config)
    checkpoint = os.path.join(outmodel_dir, 'Decoder_best.pt')
    if os.path.exists(checkpoint):
        model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    params = [
        {"params": model.parameters(), "lr": config.learning_rate, 'notchange': False}, ]
    optimizer = optim.AdamW(params, eps=1e-8)
    model, optimizer = accelerator.prepare(model, optimizer)
    accelerator.register_for_checkpointing(model)
    accelerator.register_for_checkpointing(optimizer)
    # scheduler = PolynomialLRDecay(optimizer, max_decay_steps=10000, end_learning_rate=0.000)
    if use_cuda:
        print('using GPU')
        model = model.to(device)
    # assert_vocab = AssertionVocab()
    # train_file = config.train_file
    # if not os.path.exists(train_file):
    #     prepare_data(assert_vocab)
    ori_train = readpickle(os.path.join(config.data_dir, 'processtraindata%d.pkl' % accelerator.process_index))
    train = ori_train  # [inst for inst in ori_train if len(inst.local_vocab) < 3000]
    traindataset = SumDataset(config, train)
    print(len(train))
    step_per_epoch = int(len(train) / config.train_batch_size)
    if accelerator.is_main_process:
        ori_dev = readpickle(os.path.join(config.data_dir, 'processdevdata.pkl'))
        devset = SumDataset(config, ori_dev)
    print('%d Steps for each epoch.' % (step_per_epoch))
    global_step = 0
    best_acc = -1
    best_loss = np.inf
    for epoch in range(config.train_iters):
        print('Epoch : %d' % epoch)
        epoch_start = time.time()
        step = 0
        dataloader = torch.utils.data.DataLoader(traindataset, batch_size=config.train_batch_size, shuffle=True,
                                                 num_workers=0, collate_fn=rs_collate_fn_t5)
        for batch_inst in tqdm(dataloader):
            model.train()
            step += 1
            batch_start = time.time()
            if use_cuda:
                for x in batch_inst:
                    batch_inst[x] = batch_inst[x].to(device)
            loss = model(batch_inst, train=True)
            loss = torch.sum(loss)
            nums = torch.sum(batch_inst['res'].ne(0))
            loss = loss / nums
            loss_value = loss.item()
            loss = loss / config.accumulation_steps
            if step % config.accumulation_steps == 0:
                accelerator.backward(loss)
            else:
                with accelerator.no_sync(model):
                    accelerator.backward(loss)
            global_step += 1
            batch_inst = None
            tinst = None
            torch.cuda.empty_cache()
            if accelerator.is_main_process:
                wandb.log({"average loss per step": loss_value})
            if step % config.accumulation_steps == 0:
                # update params
                optimizer.step()
                if accelerator.is_main_process:
                    lr = optimizer.state_dict()['param_groups'][0]['lr']
                    wandb.log({"learning_rate": lr})
                # update lr
                # scheduler.step()
                optimizer.zero_grad()
            if global_step % config.validate_every == 0 and accelerator.is_main_process:
                print('Start validating.')
                start = time.time()
                acc, loss, cnum = val(model, config, devset)
                end = time.time()
                print(
                    'Validation result: Acc: %.6f, loss: %.6f, time: %.2f' % (acc, loss, end - start))
                wandb.log({'validation accuracy': acc, 'cnum': cnum})
                if best_loss > loss:
                    unwrapped_model = accelerator.unwrap_model(model)
                    torch.save(unwrapped_model.state_dict(),
                               os.path.join(outmodel_dir,
                                            'Decoder_best.pt'))
                    wandb.save(os.path.join(outmodel_dir,
                                            'Decoder_best.pt'))
                    best_loss = loss
                    best_acc = acc

            if step % config.save_every == 0 and accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save(unwrapped_model.state_dict(),
                           os.path.join(outmodel_dir, 'Decoder_last.pt'))
                wandb.save(os.path.join(outmodel_dir, 'Decoder_last.pt'))
                accelerator.save_state(outmodel_dir)
            batch_end = time.time()
            if step % 100 == 0:
                print('Step %d: Loss %.4f , time %.2f' % (step, loss_value, batch_end - batch_start))

        epoch_end = time.time()
        torch.save(model.state_dict(),
                   os.path.join(outmodel_dir, 'Decoder_last.pt'))
        print('Epoch %d end in %.2f.' % (epoch, epoch_end - epoch_start))
        pass


def _train_local_debug():
    device = 0
    print('Start training.')
    model = DeepOracle_With_T5_Base(config)
    # model.load_state_dict(torch.load(os.path.join(outmodel_dir,
    #                                               'Decoder_best.pt'), map_location='cpu'))
    params = [
        {"params": model.parameters(), "lr": config.learning_rate, 'notchange': False}, ]
    optimizer = optim.AdamW(params, eps=1e-8)
    # model, optimizer = accelerator.prepare(model, optimizer)
    # accelerator.register_for_checkpointing(model)
    # accelerator.register_for_checkpointing(optimizer)
    # scheduler = PolynomialLRDecay(optimizer, max_decay_steps=10000, end_learning_rate=0.000)
    if use_cuda:
        print('using GPU')
        model = model.to(device)
    ori_train = readpickle(os.path.join(config.data_dir, 'processtraindata0.pkl'), True)
    train = ori_train  # [inst for inst in ori_train if len(inst.local_vocab) < 3000]
    traindataset = SumDataset(config, train)
    print(len(train))
    step_per_epoch = int(len(train) / config.train_batch_size)
    ori_dev = readpickle(os.path.join(config.data_dir, 'processdevdata.pkl'), True)
    devset = SumDataset(config, ori_dev)
    print('%d Steps for each epoch.' % (step_per_epoch))
    global_step = 0
    best_acc = -1
    best_loss = np.inf
    for epoch in range(config.train_iters):
        print('Epoch : %d' % epoch)
        epoch_start = time.time()
        step = 0
        dataloader = torch.utils.data.DataLoader(traindataset, batch_size=config.train_batch_size, shuffle=True,
                                                 num_workers=0, collate_fn=rs_collate_fn_t5)
        for batch_inst in tqdm(dataloader):
            model.train()
            step += 1
            batch_start = time.time()
            if use_cuda:
                for x in batch_inst:
                    batch_inst[x] = batch_inst[x].to(device)
            loss = model(batch_inst, train=True)
            loss = torch.sum(loss)
            nums = torch.sum(batch_inst['res'].ne(0))
            loss = loss / nums
            loss_value = loss.item()
            loss = loss / config.accumulation_steps
            if step % config.accumulation_steps == 0:
                loss.backward()
            global_step += 1
            batch_inst = []
            tinst = None
            torch.cuda.empty_cache()
            if step % config.accumulation_steps == 0:
                # update params
                optimizer.step()
                lr = optimizer.state_dict()['param_groups'][0]['lr']
                # update lr
                # scheduler.step()
                optimizer.zero_grad()
            if global_step % config.validate_every == 0:
                print('Start validating.')
                start = time.time()
                acc, loss, cnum = val(model, config, devset)
                end = time.time()
                print(
                    'Validation result: Acc: %.6f, loss: %.6f, time: %.2f' % (acc, loss, end - start))
                wandb.log({'validation accuracy': acc, 'cnum': cnum})
                if best_loss > loss:
                    torch.save(model.state_dict(),
                               os.path.join(outmodel_dir,
                                            'Decoder_best.pt'))
                    best_loss = loss
                    best_acc = acc

            if step % config.save_every == 0 and accelerator.is_main_process:
                torch.save(model.state_dict(),
                           os.path.join(outmodel_dir, 'Decoder_last.pt'))
            batch_end = time.time()
            if step % 100 == 0:
                print('Step %d: Loss %.4f , time %.2f' % (step, loss_value, batch_end - batch_start))

        epoch_end = time.time()
        torch.save(model.state_dict(),
                   os.path.join(outmodel_dir, 'Decoder_last.pt'))
        print('Epoch %d end in %.2f.' % (epoch, epoch_end - epoch_start))
        pass


def val(model, config, devset):
    with torch.no_grad():
        model.eval()
        accs = []
        losses = []
        cnums = []
        step = 0
        step_per_epoch = int(len(devset) / config.val_batch_size)
        print('%d Steps for each epoch.' % (step_per_epoch))
        dataloader = torch.utils.data.DataLoader(devset, batch_size=config.val_batch_size, shuffle=False, num_workers=0,
                                                 collate_fn=rs_collate_fn_t5)
        for batch_inst in dataloader:
            batch_start = time.time()
            if use_cuda:
                for x in batch_inst:
                    batch_inst[x] = batch_inst[x].to(device)
            loss, probs = model(batch_inst)
            cur_step_loss = loss.sum()
            # wandb.log({'validation loss': cur_step_loss})
            pred = probs.argmax(dim=-1)
            resmask = batch_inst['res'].ne(0)
            cur_step_loss = cur_step_loss / resmask.sum()
            losses.append(cur_step_loss.item())
            acc = (torch.eq(pred, batch_inst['res']) * resmask).float()
            resTruelen = torch.sum(resmask, dim=-1).float()
            cnum = torch.eq(resTruelen, torch.sum(acc, dim=-1))
            cnums.append(cnum.sum().item())
            acc = acc.sum(dim=-1) / resTruelen
            accs.append(acc.mean().item())
            step += 1
            duration = time.time() - batch_start
            if step % 100 == 0:
                print('Validation step %d, time %.2f' % (step, duration))
            batch_inst.clear()
        acc = np.mean(accs)
        avg_loss = np.mean(losses)
        cnum = np.sum(cnums)
        return acc, avg_loss, cnum


card = [0, 1]


def test(curid=-1):
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    device = torch.device('cuda:%d' % card[curid])
    # device = torch.device('cuda:1')
    # wandb.init(project="DeepOracle_Test_Beam=" + str(config.beam_size))
    print('Start Testing.')
    test_start = time.time()
    save_dir = os.path.join(config.generation_dir, "Beam-" + str(config.beam_size))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ground_truth_writer = open(os.path.join(save_dir, 'ground_truth%d.txt' % curid), 'w', encoding='iso8859-1')
    predicted_writer = open(os.path.join(save_dir, 'predictions_' + str(config.beam_size) + '_%d' % curid) + '.txt',
                            'w',
                            encoding='iso8859-1')
    model = DeepOracle_With_T5_Base(config)

    # params = [
    #     {"params": model.parameters(), "lr": config.learning_rate, 'notchange': False}, ]
    # optimizer = optim.AdamW(params, eps=1e-8)
    # model, optimizer = accelerator.prepare(model, optimizer)
    # accelerator.register_for_checkpointing(model)
    # accelerator.register_for_checkpointing(optimizer)
    # accelerator.load_state(outmodel_dir)
    # model = accelerator.unwrap_model(model)

    model.load_state_dict(torch.load(os.path.join(outmodel_dir, 'Decoder_best.pt'), map_location=device))
    if use_cuda:
        model = model.cuda(device)
    # assertion_vocab = AssertionVocab()
    # test_file = config.test_file
    # if not os.path.exists(test_file):
    #     prepare_data(assertion_vocab)
    test = readpickle(os.path.join(config.data_dir, 'processtestdata.pkl'))
    model.eval()
    chunk_size = len(test) // len(card) + 1
    test = test[chunk_size * curid: chunk_size * (curid + 1)]
    testset = SumDataset(config, test)
    total = len(test)
    corrected = 0
    step = 0
    total_steps = (len(test) / config.test_batch_size) + 1
    print('Total %d steps' % total_steps)
    dataloader = torch.utils.data.DataLoader(testset, batch_size=config.test_batch_size, shuffle=False, num_workers=4,
                                             collate_fn=rs_collate_fn_t5)
    testvocabs = pickle.load(open(os.path.join(config.data_dir, 'testvocab.pkl'), 'rb'))
    testassertions = pickle.load(open(os.path.join(config.data_dir, 'testassertion.pkl'), 'rb'))
    testvocabs = testvocabs[chunk_size * curid: chunk_size * (curid + 1)]
    testassertions = testassertions[chunk_size * curid: chunk_size * (curid + 1)]
    ans = []
    smoothie = SmoothingFunction().method4
    scores = []
    for i, batch_inst in enumerate(tqdm(dataloader)):
        start = time.time()
        vocabs = testvocabs[i * config.test_batch_size: (i + 1) * config.test_batch_size]
        assertions = testassertions[i * config.test_batch_size: (i + 1) * config.test_batch_size]
        batch_id_2_beams = BeamSearch(model, batch_inst, vocabs, 5, assertions, config, device)
        for assertion, (_, beams) in zip(assertions, batch_id_2_beams.items()):
            found = False
            gt_assertion = ' '.join(assertion)
            start_writing = False
            try:
                ground_truth_writer.write(gt_assertion + '\n')
                # assert len(beams) == config.beam_size
                bleu_score = sentence_bleu([assertion], beams[0].ans, smoothing_function=smoothie)
                scores.append(bleu_score)
                if not start_writing: start_writing = True
                for idx, gen_inst in enumerate(beams):
                    gen_assertion = ' '.join(gen_inst.ans)
                    if hash(gt_assertion) == hash(gen_assertion):
                        if not found:
                            ans.append(idx)
                        found = True
                    predicted_writer.write(gen_assertion + '\n')

            except Exception as e:
                traceback.print_exc()
            finally:
                if not start_writing: predicted_writer.write('Have not started wrting now\n')
                predicted_writer.write('\n')
                predicted_writer.flush()
                ground_truth_writer.flush()
                if found:
                    corrected += 1

        # wandb.log({"corrected": corrected})
        print('Step %d, #corrected = %d , estimated time remaining: %.2f' % (
            step, corrected, (time.time() - start) * (total_steps - step - 1)
        ))
        print('BLEU Score: %.4f' % np.mean(scores))
        print('top1: %.4f' % (ans.count(0) / len(scores)))
        print('top3: %.4f' % ((ans.count(0) + ans.count(1) + ans.count(2)) / len(scores)))
        step += 1
    print('BlEU Score: %.4f' % np.mean(scores))
    print('top1: %.4f' % (ans.count(0)))
    print('top3: %.4f' % (ans.count(0) + ans.count(1) + ans.count(2)))
    print('Overall acc: %d / %d = %.6f' % (corrected, total, 100 * corrected / total))
    predicted_writer.close()
    ground_truth_writer.close()
    print('Testing end in %.2f' % (time.time() - test_start))
    open('rank%d.txt' % curid, 'w').write(str(ans))
    open('bleu%d.txt' % curid, 'w').write(str(scores))


if __name__ == '__main__':
    # idx = int(sys.argv[1])
    # test(idx)

    train()
    # _train_local_debug()

    # ori_train = readpickle('datasets/processtraindata%d.pkl' % 0)
    # train = ori_train  # [inst for inst in ori_train if len(inst.local_vocab) < 3000]
    # traindataset = SumDataset(config, train)
    # print(traindataset)
