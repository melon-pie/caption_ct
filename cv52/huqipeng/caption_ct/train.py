from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import time
import os
from six.moves import cPickle

import opts
import models
from dataloader import *
import eval_utils
import misc.utils as utils
from tensorboardX import SummaryWriter
try:
    import tensorflow as tf
except ImportError:
    print("Tensorflow not installed; No tensorboard logging.")
    tf = None

# def add_summary_value(writer, key, value, iteration):
#     if writer:
#         writer.add_scalar(key, value, iteration)

def train(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    opt.use_att = utils.if_use_att(opt.caption_model)
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length
    # tensorboard logger
    tb_summary_writer = SummaryWriter(opt.checkpoint_path)

    infos = {}
    histories = {}
    if opt.start_from is not None:
        # open old infos and check if models are compatible
        with open(os.path.join(opt.start_from, 'infos_'+opt.id+'.pkl'),'rb') as f:
            infos = cPickle.load(f)
            saved_model_opt = infos['opt']
            need_be_same=["caption_model", "rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme

        if os.path.isfile(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')):
            with open(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl'),'rb') as f:
                histories = cPickle.load(f)

    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)

    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})

    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_ix = infos.get('split_ix', loader.split_ix)
    if opt.load_best_score == 1:
        best_b1 = infos.get('best_b1', None)
        best_b2 = infos.get('best_b2', None)
        best_b3 = infos.get('best_b3', None)
        best_b4 = infos.get('best_b4', None)
        best_m = infos.get('best_m', None)
        best_r_l = infos.get('best_r_l', None)
        best_val_score = infos.get('best_val_score', None)

    model = models.setup(opt)
    model.cuda()

    update_lr_flag = True
    # Assure in training mode
    model.train()

    crit = utils.LanguageModelCriterion()

    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)

    # Load the optimizer
    if vars(opt).get('start_from', None) is not None:
        optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer.pth')))

    while True:
        if update_lr_flag:
                # Assign the learning rate
            if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                decay_factor = opt.learning_rate_decay_rate  ** frac
                opt.current_lr = opt.learning_rate * decay_factor
                utils.set_lr(optimizer, opt.current_lr) # set the decayed rate
            else:
                opt.current_lr = opt.learning_rate
            # Assign the scheduled sampling prob
            if iteration > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                frac = (iteration - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
                model.ss_prob = opt.ss_prob
            update_lr_flag = False
                
        start = time.time()
        # Load data from train split (0)
        data = loader.get_batch('train')
        print('Read data:', time.time() - start)

        torch.cuda.synchronize()
        start = time.time()

        tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks']]
        # tmp = [Variable(torch.from_numpy(_), requires_grad=False).cuda() for _ in tmp]
        with torch.no_grad():
            tmp = [Variable(torch.from_numpy(_)).cuda() for _ in tmp]
        fc_feats, att_feats, labels, masks = tmp
        
        optimizer.zero_grad()
        loss = crit(model(fc_feats, att_feats, labels), labels[:,1:], masks[:,1:])
        loss.backward()
        utils.clip_gradient(optimizer, opt.grad_clip)
        optimizer.step()
        train_loss = loss.item()
        torch.cuda.synchronize()
        end = time.time()
        print("iter {} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
            .format(iteration, epoch, train_loss, end - start))
        # Update the iteration and epoch
        iteration += 1
        if data['bounds']['wrapped']:
            epoch += 1
            update_lr_flag = True

        # Write the training loss summaryCOCOEvalCap
        if (iteration % opt.losses_log_every == 0):
            tb_summary_writer.add_scalar('train_loss', train_loss, iteration)

            loss_history[iteration] = train_loss
            lr_history[iteration] = opt.current_lr
            ss_prob_history[iteration] = model.ss_prob

            tb_summary_writer.add_scalar('learning_rate', opt.current_lr, iteration)
            tb_summary_writer.add_scalar('scheduled_sampling_prob', model.ss_prob, iteration)

        # make evaluation on validation set, and save model
        if (iteration % opt.save_checkpoint_every == 0):
            # eval model
            eval_kwargs = {'split': 'val',
                            'dataset': opt.input_json}
            eval_kwargs.update(vars(opt))
            val_loss, predictions, lang_stats = eval_utils.eval_split(model, crit, loader, eval_kwargs)

            # Write validation result into summary
            tb_summary_writer.add_scalar('validation loss', val_loss, iteration)
            if lang_stats is not None:
                for k, v in lang_stats.items():
                    tb_summary_writer.add_scalar(k, v, iteration)
            val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

            # Save model and scores if is improving on validation result
            if opt.language_eval == 1:
                current_b1 = lang_stats['Bleu_1']
                current_b2 = lang_stats['Bleu_2']
                current_b3 = lang_stats['Bleu_3']
                current_b4 = lang_stats['Bleu_4']
                current_m = lang_stats['METEOR']
                current_r_l = lang_stats['ROUGE_L']
                current_score = lang_stats['CIDEr']
            else:
                current_score = - val_loss

            best_flag = False
            if True: # if true
                path = 'scores/ ' + opt.id
                if not os.path.exists(path):
                    os.mkdir(path)
                if  opt.language_eval == 1:
                    if best_b1 is None or current_b1 > best_b1:
                        best_b1 = current_b1
                        print('刷新最高b1:',best_b1)
                        # 最高评分写入文件
                        with open(path+'/best_b1.txt', 'a') as f1:
                            f1.write('epoch '+str(epoch)+'刷新最高b1:'+str(best_b1))
                            f1.write('\r\n')
                    if best_b2 is None or current_b2 > best_b2:
                        best_b2 = current_b2
                        print('刷新最高b2:',best_b2)
                        with open(path+'/best_b2.txt', 'a') as f2:
                            f2.write('epoch '+str(epoch)+'刷新最高b2:'+str(best_b2))
                            f2.write('\r\n')
                    if best_b3 is None or current_b3 > best_b3:
                        best_b3 = current_b3
                        print('刷新最高b3:',best_b3)
                        with open(path+'/best_b3.txt', 'a') as f3:
                            f3.write('epoch '+str(epoch)+'刷新最高b3:'+str(best_b2))
                            f3.write('\r\n')
                    if best_b4 is None or current_b4 > best_b4:
                        best_b4 = current_b4
                        print('刷新最高b4:',best_b4)
                        with open(path+'/best_b4.txt', 'a') as f4:
                            f4.write('epoch '+str(epoch)+'刷新最高b4:'+str(best_b4))
                            f4.write('\r\n')
                    if best_m is None or current_m > best_m:
                        best_m = current_m
                        print('刷新最高m:',best_m)
                        with open(path+'/best_m.txt', 'a') as fm:
                            fm.write('epoch '+str(epoch)+'刷新最高m:'+str(best_m))
                            fm.write('\r\n')
                    if best_r_l is None or current_r_l > best_r_l:
                        best_r_l = current_r_l
                        print('刷新最高r_l:',best_r_l)
                        with open(path+'/best_r_l.txt', 'a') as fr:
                            fr.write('epoch '+str(epoch)+'刷新最高r_l:'+str(best_r_l))
                            fr.write('\r\n')
                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    print('刷新最高cider:',best_val_score)
                    with open(path+'/scores.txt', 'a') as f:
                        f.write('epoch ' + str(epoch) + '刷新最高cider,此时各项评分为\r\n')
                        for method, score in lang_stats.items():
                            if method is 'error':
                                continue
                            f.write(method + ':' + str(score) + '\r\n')
                        f.write('\r\n')
                    best_flag = True

                if not os.path.exists(opt.checkpoint_path):
                    os.mkdir(opt.checkpoint_path)
                checkpoint_path = os.path.join(opt.checkpoint_path, 'model.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print("model saved to {}".format(checkpoint_path))
                optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer.pth')
                torch.save(optimizer.state_dict(), optimizer_path)

                # Dump miscalleous informations
                infos['iter'] = iteration
                infos['epoch'] = epoch
                infos['iterators'] = loader.iterators
                infos['split_ix'] = loader.split_ix
                infos['best_b1'] = best_b1
                infos['best_b2'] = best_b2
                infos['best_b3'] = best_b3
                infos['best_b4'] = best_b4
                infos['best_m'] = best_m
                infos['best_r_l'] = best_r_l
                infos['best_val_score'] = best_val_score
                infos['opt'] = opt
                infos['vocab'] = loader.get_vocab()

                histories['val_result_history'] = val_result_history
                histories['loss_history'] = loss_history
                histories['lr_history'] = lr_history
                histories['ss_prob_history'] = ss_prob_history
                with open(os.path.join(opt.checkpoint_path,'infos_'+opt.id+'.pkl'), 'wb') as f:
                    cPickle.dump(infos, f)
                with open(os.path.join(opt.checkpoint_path,'histories_'+opt.id+'.pkl'), 'wb') as f:
                    cPickle.dump(histories, f)

                if best_flag:
                    checkpoint_path = os.path.join(opt.checkpoint_path, 'model-best.pth')
                    torch.save(model.state_dict(), checkpoint_path)
                    print("model-best saved to {}".format(checkpoint_path))
                    with open(os.path.join(opt.checkpoint_path, 'infos_'+str(best_val_score) + opt.id+'-best.pkl'), 'wb') as f:
                        cPickle.dump(infos, f)

        # Stop if reaching max epochs
        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break


opt = opts.parse_opt()
train(opt)
