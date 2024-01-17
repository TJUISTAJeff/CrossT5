import gc
import os.path
import sys
import time

import torch

sys.path.extend([".", ".."])
from CONSTANTS import *
from torch import optim
from utils.processing import data_iter, compute_accuracy, \
    generate_decoder_tensor_instance
from data.dataloader import prepare_data, data_splitter
from data.vocab import AssertionVocab
from entities.instance import Instance
from Modules.ScheduledOptim import *
from Models.DeepOracle import Decoder
from Modules.BeamSearch import BeamSearch
import javalang

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
model = Decoder(d_model=codebert_encoder.config.hidden_size)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
optimizer = ScheduledOptim(optimizer, d_model=768, n_warmup_steps=700)
if use_cuda:
    logger.info('using GPU')
    model = model.to(device)


def train():
    logger.info('Start training.')
    global_vocab = AssertionVocab()
    instances = prepare_data(global_vocab)
    train, dev, test = data_splitter(instances, [8, 1, 1])
    logger.info('Train: %d, Validation: %d, Test: %d' % (len(train), len(dev), len(test)))
    step_per_epoch = int(len(train) / train_batch_size)
    logger.info('%d Steps for each epoch.' % (step_per_epoch))
    global_step = 0
    best_acc = -1
    missed_batches = 0
    for epoch in range(100000):
        model.train()
        logger.info('Epoch : %d' % epoch)
        epoch_start = time.time()
        step = 0
        for batch_inst in data_iter(train, train_batch_size):
            # try:
            batch_start = time.time()
            batch = generate_decoder_tensor_instance(batch_inst, global_vocab)
            encoder_outputs, queries, encoder_time = encode(batch)
            if use_cuda:
                queries = queries.to(device)
                encoder_outputs = encoder_outputs.to(device)
                batch.to_cuda(device)
            # encoder_hiddens, encoder_masks, queries, query_masks, vocab_keys, vocab_values, vocab_masks
            pred = model(encoder_outputs, batch.encoder_masks, queries, batch.query_masks, batch.vocab_keys,
                         batch.vocab_mask)
            loss = model.compute_loss(pred, batch.target)
            loss /= accumulation_steps
            loss_value = loss.data.cpu().numpy()
            loss.backward()
            if step % accumulation_steps == 0:
                optimizer.step_and_update_lr()
                optimizer.zero_grad()
            batch_end = time.time()
            step += 1
            logger.info('Step %d: Loss: %.4f, time %.2f (codebert encoder %.2f), estimate time per epoch: %.2f' % (
                step, loss_value, batch_end - batch_start, encoder_time,
                step_per_epoch * (batch_end - batch_start)))
            global_step += 1
            batch.clear()
            if global_step % 5000 == 0:
                acc = val(dev, global_vocab)
                if acc > best_acc:
                    torch.save(model.state_dict(),
                               os.path.join(outmodel_dir,
                                            'Decoder_best.ckpt'))

            if step % 100 == 0:
                torch.save(model.state_dict(),
                           os.path.join(outmodel_dir, 'Decoder_last.ckpt'))
            # except RuntimeError:
            #     missed_batches += 1
            #     print('oom')
            #     torch.cuda.empty_cache()
            # finally:
            #     continue

        epoch_end = time.time()
        torch.save(model.state_dict(),
                   os.path.join(outmodel_dir, 'Decoder_last.ckpt'))
        logger.info('Epoch %d end in %.2f, %d batches are missed due to memory limit.' % (
            epoch, epoch_end - epoch_start, missed_batches))
        pass


def val(instances, global_vocab):
    model.eval()
    logger.info('Start validating')
    step_per_epoch = int(len(instances) / validate_batch_size)
    logger.info('%d Steps for each epoch.' % (step_per_epoch))
    step = 0
    with torch.no_grad():
        total = 0
        corrected = 0
        for batch_inst in data_iter(instances, validate_batch_size):
            batch_start = time.time()
            batch = generate_decoder_tensor_instance(batch_inst, global_vocab)
            encoder_outputs, queries, encoder_time = encode(batch)
            if use_cuda:
                queries = queries.to(device)
                encoder_outputs = encoder_outputs.to(device)
                batch.to_cuda(device)
            _, pred = model.predict_one(encoder_outputs, batch.encoder_masks, queries, batch.query_masks,
                                        batch.vocab_keys, batch.vocab_mask)
            corrected += compute_accuracy(pred=pred, truth=batch.target.detach().cpu().numpy(), normalize=False)
            total += len(batch.target.detach().cpu().numpy())
            batch.clear()
            gc.collect()
            step += 1
            batch_end = time.time()
            logger.info(
                'Validating Step %d: corrected: %d, time %.2f (encoder: %.2f), estimate time per epoch: %.2f' % (
                    step, corrected, batch_end - batch_start, encoder_time,
                    step_per_epoch * (batch_end - batch_start)))
    batch_end = time.time()
    print(total)
    acc = float(corrected / total)
    logger.info('Validating acc: %d/%d =  %.4f, time: %.2f' % (corrected, total, acc, batch_end - batch_start))
    return acc


def encode(batch):
    with torch.no_grad():
        if use_cuda:
            encoder_inputs = batch.encoder_inputs.to(encoder_device)
            decoder_queries = batch.decoder_queries.to(encoder_device)
        else:
            encoder_inputs = batch.encoder_inputs
            decoder_queries = batch.decoder_queries
        encoder_start = time.time()
        encoder_outputs = codebert_encoder(**encoder_inputs, output_hidden_states=True)['last_hidden_state']
        queries_output = codebert_encoder(**decoder_queries)
        encoder_inputs = None
        decoder_queries = None
        encoder_end = time.time()
        batch.clear_inputs()
        queries = queries_output['last_hidden_state']
    return encoder_outputs, queries, encoder_end - encoder_start


def test(beam_size=50):
    logger.info('Start testing.')

    ground_truth_writer = open('datasets/results/ground_truth.txt', 'w', encoding='utf-8')
    predicted_writer = open('datasets/results/prefictions_' + str(beam_size) + '.txt', 'w', encoding='utf-8')
    if not os.path.exists('datasets/results'):
        os.makedirs('datasets/results')

    model.load_state_dict(torch.load(os.path.join(outmodel_dir, 'Decoder_best.ckpt')))
    global_vocab = AssertionVocab()
    instances = prepare_data(global_vocab)
    _, _, test = data_splitter(instances, [8, 1, 1])
    logger.info('Init models and modules.')
    total = len(test)
    corrected = 0
    failed_parsing = 0
    predict_num = 0
    for batch_inst in data_iter(test, test_batch_size):
        b_s = BeamSearch(beam_size)
        batch_start = time.time()
        gt_instance = batch_inst[0]
        beam_instances = []
        for inst in batch_inst:
            beam_instances.append(Instance([inst.assertion[0]], inst.assert_type, inst.context_tokens, inst.local_vocab))
        outputs = b_s.generate(beam_instances, model, global_vocab)
        ground_truth = ' '.join(gt_instance.assertion)
        ground_truth_writer.write(ground_truth + '\n')
        predicted_writer.write('\n')
        key = hash(ground_truth)
        batch_end = time.time()
        logger.info('Predict in %.2f' % (batch_end - batch_start))
        for generated_assertion in outputs:
            try:
                tks = javalang.tokenizer.tokenize(generated_assertion)
                tokenized_gen_assertion = ' '.join([tk.value for tk in tks])
                predicted_writer.write(tokenized_gen_assertion + '\n')
                gen_key = hash(tokenized_gen_assertion)
                if key == gen_key:
                    corrected += 1
                    break
            except Exception:
                failed_parsing += 1
                predicted_writer.write(generated_assertion + '\n')
                pass
            finally:
                predict_num += 1
                continue
    logger.info('%d / %d = %.4f failed parsing.' % (failed_parsing, predict_num, (failed_parsing / predict_num)))
    logger.info('acc: %d/%d = %.2f' % (corrected, total, corrected / total))
    ground_truth_writer.close()
    predicted_writer.close()


if __name__ == '__main__':
    if sys.argv[1] == 'train':
        train()
    elif sys.argv[1] == 'resume':
        model.load_state_dict(torch.load(os.path.join(outmodel_dir, 'Decoder_last.ckpt')))
        train()
        pass
    # elif sys.argv[1] == 'val':
    #     global_vocab = GlobalVocab()
    #     instances = load_all_data(global_vocab)
    #     _, dev, _ = data_splitter(instances, [8, 1, 1])
    #     val(dev, global_vocab)
    elif sys.argv[1] == 'test':
        beam_size = sys.argv[2]
        if beam_size.isdecimal():
            test(int(beam_size))
        else:
            print('Illegal argument %s, required decimal.' % beam_size)
        pass
