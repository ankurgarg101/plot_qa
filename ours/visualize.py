"""
Module that visualizes attention weights on a specific image at a specified checkpoint
"""

import argparse
import numpy as np
import json
import os
import torch
import torch.nn as nn
import torchvision
import h5py

import skimage
from skimage import io

def load_models(models, params, load_model_dir):

    if 'ques_model' in models:
        models['ques_model'].load_state_dict(torch.load(
            os.path.join(load_model_dir, 'question_model.pkl')))
    
    if 'att_model' in models:
        models['att_model'].load_state_dict(torch.load(
            os.path.join(load_model_dir, 'attention_model.pkl')))

    if 'img_model' in models:
        models['img_model'].load_state_dict(torch.load(
            os.path.join(load_model_dir, 'image_model.pkl')))

    if 'roi_model' in models:
        models['roi_model'].load_state_dict(torch.load(
        os.path.join(load_model_dir, 'roi_model.pkl')))

    if 'text_model' in models:
        models['text_model'].load_state_dict(torch.load(
            os.path.join(load_model_dir, 'text_model.pkl')))

    return models

def normalize_att_image(att_image):

    min_val = np.min(att_image)
    max_val = np.max(att_image)

    scale = float(max_val - min_val)

    return (att_image - min_val) / scale

def visualize_global_img_attn(params, attn_img_emb, rgb_image, out_img_name, alpha = 0.5):
    attention_image = skimage.transform.pyramid_expand(attn_img_emb.reshape(14, 14), upscale = 32, multichannel = False).reshape(448, 448, 1)
    attention_image = normalize_att_image(attention_image)
    final_attention_image = alpha*rgb_image + (1-alpha)*attention_image
    
    out_image_path = os.path.join(params['vis_dir'], out_img_name)
    io.imsave(out_image_path, final_attention_image)
    return final_attention_image

def visualize_text_attn(dataset, idx, params, attn_text, rgb_image, out_img_name, alpha = 0.5):
    attention_image = np.zeros((448, 448))
    count = np.zeros((448, 448))
    # print (attn_text)
    qid = dataset.idx2qid[idx]
    qas = dataset.qa_dict[qid]
    print (qas)
    mt = dataset.metadata_dict[qas['image']]
    # print (mt)
    EPS = 1e-20
    ids2texts = {}
    for t in mt['texts']:
        ids2texts[t['idx']] = t

    for i in range(30):
        if abs(attn_text[0, i]) > EPS:
            bbox = ids2texts[i]['bbox']
            x, y, w, h = bbox
            ix = int(x)
            iy = int(y)
            iw = int(w)
            ih = int(h)
            attention_image[iy: iy+ih+1, ix: ix+iw+1] += attn_text[0, i]
            count[iy: iy+ih+1, ix: ix+iw+1] += 1

    count = np.maximum(1, count)
    attention_image = np.divide(attention_image, count)
    attention_image = attention_image.reshape(448, 448, 1)
    attention_image = normalize_att_image(attention_image)
    final_attention_image = alpha*rgb_image + (1-alpha)*attention_image
    
    out_image_path = os.path.join(params['vis_dir'], out_img_name)
    io.imsave(out_image_path, final_attention_image)
    return final_attention_image

def visualize_roi_attn(dataset, idx, params, attn_roi, rgb_image, out_img_name, alpha = 0.5):
    attention_image = np.zeros((448, 448))
    count = np.zeros((448, 448))
    print (attn_roi)
    qid = dataset.idx2qid[idx]
    qas = dataset.qa_dict[qid]
    print (qas)
    mt = dataset.metadata_dict[qas['image']]
    print (mt)
    EPS = 1e-20

    print(attn_roi)
    print(np.sum(attn_roi, axis=1))
    
    bidx = 0
    for br in mt['bars']['bboxes']:
        for box in br:
            if bidx < 30 and (abs(attn_roi[0, bidx]) > EPS):
                x, y, w, h = box
                ix = int(x)
                iy = int(y)
                iw = int(w)
                ih = int(h)
                attention_image[iy: iy+ih+1, ix: ix+iw+1] += attn_roi[0, bidx]
                count[iy: iy+ih+1, ix: ix+iw+1] += 1

            bidx += 1

    count = np.maximum(1, count)
    attention_image = np.divide(attention_image, count)

    attention_image = attention_image.reshape(448, 448, 1)
    attention_image = normalize_att_image(attention_image)
    final_attention_image = alpha*rgb_image + (1-alpha)*attention_image
    
    out_image_path = os.path.join(params['vis_dir'], out_img_name)
    io.imsave(out_image_path, final_attention_image)
    return final_attention_image

def visualize_model(models, dataset, params, extra_params, idx):
    
    # # Construct Data loader
    # dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=params['batch_size'], shuffle=False, num_workers=1)
    
    if params['use_gpu'] and torch.cuda.is_available():
        print('Initialized Cuda Models')
        
        for mname in models:
            models[mname] = models[mname].cuda()

    print('Loading Models')
    
    load_model_dir = os.path.join(params['checkpoint_path'])
    
    print('Loading model files from folder: %s' % load_model_dir)
    
    models = load_models(models, params, load_model_dir)

    if params['use_roi'] or params['load_roi']:
        roi_save_file = h5py.File(params['roi_save_file'])

    accuracies = []
    # Call train() on all models for training
    for m in models:
        models[m].eval()
        
    if params['use_gpu'] and torch.cuda.is_available():
        pred = lambda x: np.argmax(x.cpu().detach().numpy(), axis=1)
    else:    
        pred = lambda x: np.argmax(x.detach().numpy(), axis=1)

    indices = [idx]
    # print (indices)
    subset = torch.utils.data.Subset(dataset, indices)
    
    # dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1)
    for i, batch in enumerate(subset):

        # images = batch['image']
        # questions = batch['ques']
        # ques_lens = batch['ques_len']
        # answers = batch['ans']
        # bar_lens = batch['bar_len']
        # text_lens = batch['text_len']
        # bar_bboxes = batch['bar_bboxes']
        # text_bboxes = batch['text_bboxes']
        # text_vals = batch['text_vals']
        # text_types = batch['text_types']
        # ids = batch['id']
        # if params['load_roi']:
        #     roi_feats = batch['roi_feats']
        
        # Jugaad for batch size 1 using subset
        images = batch['image'].unsqueeze(0)
        questions = batch['ques'].unsqueeze(0)
        ques_lens = torch.LongTensor([batch['ques_len']])
        answers = batch['ans']
        bar_lens = torch.LongTensor([batch['bar_len']])
        text_lens = torch.LongTensor([batch['text_len']])
        bar_bboxes = batch['bar_bboxes'].unsqueeze(0)
        text_bboxes = batch['text_bboxes'].unsqueeze(0)
        text_vals = batch['text_vals'].unsqueeze(0)
        text_types = batch['text_types'].unsqueeze(0)
        ids = [batch['id']]
        if params['load_roi']:
            roi_feats = batch['roi_feats'].unsqueeze(0)

        # # Sort the examples in reverse order of sentence length
        # _, sort_idxes = torch.sort(ques_lens, descending=True)
        # images = images[sort_idxes, :, :, :]
        # questions = questions[sort_idxes, :]
        # ques_lens = ques_lens[sort_idxes]
        # answers = answers[sort_idxes, :]
        # answers = answers.squeeze(1)
        # bar_lens = bar_lens[sort_idxes]
        # text_lens = text_lens[sort_idxes]
        # bar_bboxes = bar_bboxes[sort_idxes]
        # text_bboxes = text_bboxes[sort_idxes]
        # text_vals = text_vals[sort_idxes]
        # text_types = text_types[sort_idxes]
        # ids = [ids[i] for i in sort_idxes]
        # if params['load_roi']:
        #     roi_feats = roi_feats[sort_idxes]
        
        # print (images)
        # print (questions)
        # print (ques_lens)
        # print (answers)

        if (params['use_gpu'] and torch.cuda.is_available()):
            images = images.cuda()
            questions = questions.cuda()
            answers = answers.cuda()
            bar_lens = bar_lens.cuda()
            text_lens = text_lens.cuda()
            bar_bboxes = bar_bboxes.cuda()
            text_bboxes = text_bboxes.cuda()
            text_vals = text_vals.cuda()
            text_types = text_types.cuda()
            ids = ids
            if params['load_roi']:
                roi_feats = roi_feats.cuda()

        ques_emb = models['ques_model'].forward(questions, ques_lens)
        
        global_img_feats = None
        if params['load_roi']:
            img_emb = roi_feats

            if params['use_global_img']:
                assert 'img_model' in models
                img_emb = models['img_model'].forward(images)
                global_img_feats = img_emb.view(img_emb.size(0), img_emb.size(1), -1).permute(0, 2, 1)

        else:
            img_emb = models['img_model'].forward(images)

            if params['use_global_img']:
                global_img_feats = img_emb.view(img_emb.size(0), img_emb.size(1), -1).permute(0, 2, 1)

            if params['use_roi']:

                box_idx = torch.as_tensor(np.repeat(range(len(images)), extra_params['max_num_bars']), dtype=torch.int)
                if (params['use_gpu'] and torch.cuda.is_available()):
                    box_idx = box_idx.cuda()

                img_emb = models['roi_model'].forward(img_emb, bar_bboxes, box_idx )
                
                # Jugaad
                img_emb = img_emb.unsqueeze(0)
                
                if params['roi_save_file']:
                    for b in range(len(ids)):
                        curr_id = ids[b]
                        curr_emb = img_emb[b,:].cpu().numpy()
                        if curr_id not in list(roi_save_file.keys()):
                            roi_save_file.create_dataset(curr_id,data=curr_emb)

            else:
                img_emb = img_emb.view(img_emb.size(0), img_emb.size(1), -1).permute(0, 2, 1)

        if params['use_text']:
            text_emb = models['text_model'].forward(text_vals)
            text_emb = torch.cat((text_emb, text_types), dim=2)
        else:
            text_emb = None
        
        if params['use_pos']:

            if params['use_text']:
                text_emb = torch.cat((text_emb, text_bboxes), dim=2)

            if params['use_roi']:
                img_emb = torch.cat((img_emb, bar_bboxes), dim=2)
        
        if params['use_roi'] or params['load_roi']:
            output = models['att_model'].forward(ques_emb, img_emb, num_boxes=bar_lens, text_feats=text_emb, num_texts=text_lens, global_img_feats=global_img_feats)
        else:
            output = models['att_model'].forward(ques_emb, img_emb, num_boxes=None, text_feats=text_emb, num_texts=text_lens, global_img_feats=global_img_feats)

        if (params['use_gpu'] and torch.cuda.is_available()):
            answers = answers.cpu()

        output_preds = pred(output)
        print ("Prediction:", output_preds)
        print ("Gold:", answers)
        print ("Pass:", output_preds == answers.detach().numpy())

        # Visualization Code
        attn_1_img_emb = models['att_model'].attn_1_img_emb.cpu().detach().numpy()
        attn_2_img_emb = models['att_model'].attn_2_img_emb.cpu().detach().numpy()

        if params['use_text']:
            attn_1_text = models['att_model'].attn_1_text.cpu().detach().numpy()
            attn_2_text = models['att_model'].attn_2_text.cpu().detach().numpy()
        
        if params['use_global_img']:
            attn_1_global_img = models['att_model'].attn_1_global_img.cpu().detach().numpy()
            attn_2_global_img = models['att_model'].attn_2_global_img.cpu().detach().numpy()

        if not os.path.exists(os.path.join(params['vis_dir'])):
            os.makedirs(os.path.join(params['vis_dir']))

        # Load the input image
        inp_image_path = os.path.join(params['data_dir'], 'images', ids[0])
        rgba_image = io.imread(inp_image_path)
        rgb_image = skimage.color.rgba2rgb(rgba_image)

        att1_final_img = np.zeros((448, 448, 3))
        att2_final_img = np.zeros((448, 448, 3))
        norm_factor = 0

        # Visualize img_emb
        if not (params['use_roi'] or params['load_roi']):
            # Attention is computed over the global image features
            out_img_name_1 = ids[0] + '_attn_1_img_emb.png'
            out_img_name_2 = ids[0] + '_attn_2_img_emb.png'
            att1_img = visualize_global_img_attn(params, attn_1_img_emb, rgb_image, out_img_name_1)
            att2_img = visualize_global_img_attn(params, attn_2_img_emb, rgb_image, out_img_name_2)
            
            att1_final_img += att1_img
            att2_final_img += att2_img
            norm_factor += 1            
        else:
            # Attention is computed over the ROI features
            if params['use_pos']:
                pos = 'pos_'
            else:
                pos = ''
            out_img_name_1 = ids[0] + '_attn_1_roi_' + pos + 'emb.png'
            out_img_name_2 = ids[0] + '_attn_2_roi_' + pos + 'emb.png'
            att1_roi_img = visualize_roi_attn(dataset, idx, params, attn_1_img_emb, rgb_image, out_img_name_1)
            att2_roi_img = visualize_roi_attn(dataset, idx, params, attn_2_img_emb, rgb_image, out_img_name_2)
            att1_final_img += att1_roi_img
            att2_final_img += att2_roi_img
            norm_factor += 1

        # Visualize text_emb
        if params['use_text']:
            if params['use_pos']:
                pos = 'pos_'
            else:
                pos = ''
            out_img_name_1 = ids[0] + '_attn_1_text_' + pos + 'emb.png'
            out_img_name_2 = ids[0] + '_attn_2_text_' + pos + 'emb.png'
            att1_text_img = visualize_text_attn(dataset, idx, params, attn_1_text, rgb_image, out_img_name_1)
            att2_text_img = visualize_text_attn(dataset, idx, params, attn_2_text, rgb_image, out_img_name_2)
            att1_final_img += att1_text_img
            att2_final_img += att2_text_img
            norm_factor += 1

        # Visualize global_img_emb
        if params['use_global_img']:
            # Attention is computed over the global image features
            out_img_name_1 = ids[0] + '_attn_1_global_img_emb.png'
            out_img_name_2 = ids[0] + '_attn_2_global_img_emb.png'
            att1_global_img = visualize_global_img_attn(params, attn_1_global_img, rgb_image, out_img_name_1)
            att2_global_img = visualize_global_img_attn(params, attn_2_global_img, rgb_image, out_img_name_2)
            att1_final_img += att_1_global_img
            att2_final_img += att2_global_img
            norm_factor += 1

        out_image_path = os.path.join(params['vis_dir'], '{}_{}_att1_final.png'.format(ids[0], idx))
        io.imsave(out_image_path, normalize_att_image(att1_final_img))

        out_image_path = os.path.join(params['vis_dir'], '{}_{}_att2_final.png'.format(ids[0], idx))
        io.imsave(out_image_path, normalize_att_image(att2_final_img))

        break
    