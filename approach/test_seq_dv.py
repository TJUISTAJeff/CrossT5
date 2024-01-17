import os.path
import sys

from Modules.BeamSearch import BeamSearch

sys.path.extend([".", ".."])
from CONSTANTS import *
from Models.SeqDv import SeqDv
from torch import optim
# from data.dataloader import prepare_data
# from data.vocab import AssertionVocab
# from Models.DeepOracle import DeepOracle
# from Modules.BeamSearch import BeamSearch
import wandb
# from accelerate import Accelerator
from Modules.SumDataset import *
import torch

# accelerator = Accelerator()
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
config = Configurable(os.path.join(PROJECT_ROOT, 'config/default.ini'))
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5-small')
from tqdm import tqdm


# CharVocab()
def pad_seq(seq, maxlen):
    act_len = len(seq)
    if len(seq) < maxlen:
        seq = seq + [0] * maxlen
        seq = seq[:maxlen]
    else:
        seq = seq[:maxlen]
        act_len = maxlen
    return seq


device = None


def train():
    global device
    logger.info('Start training.')
    model = SeqDv(config)
    # model.load_state_dict(torch.load(os.path.join(outmodel_dir,
    #                                               'Decoder_best.ckpt'), map_location='cpu'))
    params = [
        {"params": model.parameters(), "lr": config.learning_rate, 'notchange': False}, ]
    optimizer = optim.AdamW(params, eps=1e-8)
    # scheduler = PolynomialLRDecay(optimizer, max_decay_steps=10000, end_learning_rate=0.000)
    if use_cuda and torch.cuda.is_available():
        device = 'cuda'
        logger.info('using GPU')
        model = model.to(device)
    # assert_vocab = AssertionVocab()
    # train_file = config.train_file
    # if not os.path.exists(train_file):
    #     prepare_data(assert_vocab)
    ori_train = readpickle(os.path.join(PROJECT_ROOT, f'datasets/processtraindata0.pkl'))
    # ori_train = readpickle(os.path.join(PROJECT_ROOT, 'datasets/processdevdata.pkl'))#, True)
    # ori_train = readpickle(os.path.join(PROJECT_ROOT, 'datasets/processtestdata.pkl'))
    train = ori_train  # [inst for inst in ori_train if len(inst.local_vocab) < 3000]
    traindataset = SumDataset(config, train)
    print(len(train))
    step_per_epoch = int(len(train) / config.train_batch_size)
    # if accelerator.is_main_process:
    #     ori_dev = readpickle(os.path.join(PROJECT_ROOT, 'datasets/processdevdata.pkl'))
    ori_dev = readpickle(os.path.join(PROJECT_ROOT, f'datasets/processtraindata0.pkl'))
        # devset = SumDataset(config, ori_dev)
    devset = SumDataset(config, ori_dev)
    logger.info('%d Steps for each epoch.' % (step_per_epoch))
    global_step = 0
    best_acc = -1
    best_loss = np.inf
    for epoch in range(config.train_iters):
        logger.info('Epoch : %d' % epoch)
        epoch_start = time.time()
        step = 0
        # dataloader = torch.utils.data.DataLoader(traindataset, batch_size=config.train_batch_size, shuffle=True,
        #                                          num_workers=4, collate_fn=rs_collate_fn)
        dataloader = torch.utils.data.DataLoader(traindataset, batch_size=config.train_batch_size, shuffle=True, collate_fn=rs_collate_fn)
        for batch_inst in tqdm(dataloader):
            model.train()
            step += 1
            batch_start = time.time()
            if use_cuda:
                for x in batch_inst:
                    batch_inst[x] = batch_inst[x].to(device)
            loss = model(batch_inst, train=True)
            print("after getting loss")
            for name, p in model.named_parameters():
                #  print(name)
                if p.grad is None:
                    print(name)
            loss = torch.sum(loss)
            nums = torch.sum(batch_inst['res'].ne(0))
            loss = loss / nums
            loss_value = loss.item()
            loss = loss / config.accumulation_steps
            if step % config.accumulation_steps == 0:
                loss.backward()
            #     accelerator.backward(loss)
            # else:
            #     with accelerator.no_sync(model):
            #         accelerator.backward(loss)
            global_step += 1
            batch_inst.clear()
            # if accelerator.is_main_process:
            #     wandb.log({"average loss per step": loss_value})
            tinst = None
            if step % config.accumulation_steps == 0:
                # update params
                optimizer.step()
                print("update params")
                for name, p in model.named_parameters():
                    #  print(name)
                    if p.grad is None:
                        print(name)

                # if accelerator.is_main_process:
                #     lr = optimizer.state_dict()['param_groups'][0]['lr']
                #     wandb.log({"learning_rate": lr})
                # update lr
                # scheduler.step()
                optimizer.zero_grad()
            if global_step % config.validate_every == 0 : #and accelerator.is_main_process:
                logger.info('Start validating.')
                start = time.time()
                acc, loss, cnum = val(model, config, devset)
                end = time.time()
                logger.info(
                    'Validation result: Acc: %.6f, loss: %.6f, time: %.2f' % (acc, loss, end - start))
                # wandb.log({'validation accuracy': acc, 'cnum': cnum})
                # if best_loss > loss:
                #     unwrapped_model = accelerator.unwrap_model(model)
                #     torch.save(unwrapped_model.state_dict(),
                #                os.path.join(outmodel_dir,
                #                             'Decoder_best.ckpt'))
                #     best_loss = loss
                #     best_acc = acc

            # if step % config.save_every == 0 and accelerator.is_main_process:
            #     unwrapped_model = accelerator.unwrap_model(model)
            #     torch.save(unwrapped_model.state_dict(),
            #                os.path.join(outmodel_dir, 'Decoder_last.pt'))
            #     accelerator.save_state(outmodel_dir)
            batch_end = time.time()
            if step % 100 == 0:
                logger.info('Step %d: Loss %.4f , time %.2f' % (step, loss_value, batch_end - batch_start))

        epoch_end = time.time()
        torch.save(model.state_dict(),
                   os.path.join(outmodel_dir, 'Decoder_last.pt'))
        logger.info('Epoch %d end in %.2f.' % (epoch, epoch_end - epoch_start))
        pass


def val(model, config, devset):
    with torch.no_grad():
        model.eval()
        accs = []
        losses = []
        cnums = []
        step = 0
        step_per_epoch = int(len(devset) / config.val_batch_size)
        logger.info('%d Steps for each epoch.' % (step_per_epoch))
        # dataloader = torch.utils.data.DataLoader(devset, batch_size=config.val_batch_size, shuffle=False, num_workers=4,
        #                                          collate_fn=rs_collate_fn)
        dataloader = torch.utils.data.DataLoader(devset, batch_size=config.val_batch_size, shuffle=False, collate_fn=rs_collate_fn)
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
                logger.info('Validation step %d, time %.2f' % (step, duration))
            batch_inst.clear()

        acc = np.mean(accs)
        avg_loss = np.mean(losses)
        cnum = np.sum(cnums)
        return acc, avg_loss, cnum


# card = [0, 1]
card = [0]

def test(curid=-1):
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    device = torch.device('cuda:%d' % card[curid])
    # user = "iista_oracle_team"
    # project = "DeepOracle_Test_Beam=" + str(config.beam_size)
    # run_id = "ablation_seq2seq_test_1"
    # wandb.init(entity=user, project=project, name=run_id)
    # wandb.init(project="DeepOracle_Test_Beam=" + str(config.beam_size))
    logger.info('Start Testing.')
    test_start = time.time()
    save_dir = os.path.join(os.path.join(PROJECT_ROOT, config.generation_dir), "Beam-" + str(config.beam_size))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ground_truth_writer = open(os.path.join(save_dir, 'ground_truth%d.txt' % curid), 'w', encoding='iso8859-1')
    predicted_writer = open(os.path.join(save_dir, 'predictions_' + str(config.beam_size) + '_%d' % curid) + '.txt',
                            'w',
                            encoding='iso8859-1')
    model = SeqDv(config)
    model.load_state_dict(torch.load(os.path.join(outmodel_dir, 'Decoder_best.ckpt'), map_location=device))
    if use_cuda:
        model = model.cuda(device)
    # assertion_vocab = AssertionVocab()
    # test_file = config.test_file
    # if not os.path.exists(test_file):
    #     prepare_data(assertion_vocab)
    test = readpickle(os.path.join(PROJECT_ROOT, 'datasets/processtestdata.pkl'))
    model.eval()
    chunk_size = len(test) // len(card) + 1
    test = test[chunk_size * curid: chunk_size * (curid + 1)]
    testset = SumDataset(config, test)
    total = len(test)
    corrected = 0
    step = 0
    total_steps = (len(test) / config.test_batch_size) + 1
    logger.info('Total %d steps' % total_steps)
    # dataloader = torch.utils.data.DataLoader(testset, batch_size=config.test_batch_size, shuffle=False, num_workers=4,
    #                                          collate_fn=rs_collate_fn)
    dataloader = torch.utils.data.DataLoader(testset, batch_size=config.test_batch_size, shuffle=False,
                                             collate_fn=rs_collate_fn)
    testvocabs = pickle.load(open(os.path.join(PROJECT_ROOT, 'datasets/testvocab.pkl'), 'rb'))
    testassertions = pickle.load(open(os.path.join(PROJECT_ROOT, 'datasets/testassertion.pkl'), 'rb'))
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
            try:
                ground_truth_writer.write(gt_assertion + '\n')
                # assert len(beams) == config.beam_size
                bleu_score = sentence_bleu([assertion], beams[0].ans, smoothing_function=smoothie)
                scores.append(bleu_score)
                for idx, gen_inst in enumerate(beams):
                    gen_assertion = ' '.join(gen_inst.ans)
                    if hash(gt_assertion) == hash(gen_assertion):
                        if not found:
                            ans.append(idx)
                        found = True
                    predicted_writer.write(gen_assertion + '\n')
                predicted_writer.write('\n')
                predicted_writer.flush()
                ground_truth_writer.flush()
                if found:
                    corrected += 1
            except Exception as e:
                total -= 1
        # wandb.log({"corrected": corrected})
        logger.info('Step %d, #corrected = %d , estimated time remaining: %.2f' % (
            step, corrected, (time.time() - start) * (total_steps - step - 1)
        ))
        logger.info('BLEU Score: %.4f' % np.mean(scores))
        logger.info('top1: %.4f' % (ans.count(0) / len(scores)))
        logger.info('top3: %.4f' % ((ans.count(0) + ans.count(1) + ans.count(2)) / len(scores)))
        step += 1
    logger.info('BlEU Score: %.4f' % np.mean(scores))
    logger.info('top1: %.4f' % (ans.count(0)))
    logger.info('top3: %.4f' % (ans.count(0) + ans.count(1) + ans.count(2)))
    logger.info('Overall acc: %d / %d = %.6f' % (corrected, total, 100 * corrected / total))
    predicted_writer.close()
    ground_truth_writer.close()
    logger.info('Testing end in %.2f' % (time.time() - test_start))
    open('rank%d.txt' % curid, 'w').write(str(ans))
    open('bleu%d.txt' % curid, 'w').write(str(scores))


if __name__ == '__main__':
    # idx = int(sys.argv[1])
    # test(idx)

    # train()
    test(0)
    # ori_train = readpickle('datasets/processtraindata%d.pkl' % 0)
    # train = ori_train  # [inst for inst in ori_train if len(inst.local_vocab) < 3000]
    # traindataset = SumDataset(config, train)
