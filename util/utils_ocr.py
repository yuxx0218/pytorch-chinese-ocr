# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
from PIL import Image
import json
import cv2

import torch

from config import GPU



def transcription(t, charactersPred):
    _, t = t.transpose(0, 1).max(0)
    t = t.contiguous().view(-1).cpu().numpy()

    length = len(t)
    char_list = []
    n = len(charactersPred)
    for i in range(length):
        if t[i] not in [n - 1, n - 1] and (not (i > 0 and t[i - 1] == t[i])):
            char_list.append(charactersPred[t[i]])
    return ''.join(char_list)


def transcription_dict(pred, charactersPred):
    pred = pred.data.cpu().numpy()
    t = pred.argmax(axis=1)
    prob = [2**pred[ind, pb] for ind, pb in enumerate(t)]

    length = len(t)
    charList = []
    probList = []
    n = len(charactersPred)
    for i in range(length):
        if t[i] not in [n - 1, n - 1] and (not (i > 0 and t[i - 1] == t[i])):
            charList.append(charactersPred[t[i]])
            probList.append(prob[i])
    res = {'text': ''.join(charList),
           "prob": round(float(min(probList)), 2) if len(probList) > 0 else 0,
           "chars": [{'char': char, 'prob': round(float(p), 2)} for char, p in zip(charList, probList)]}
    return res


def predict(model, image, alphabet):
    scale = image.size[1] * 1.0 / 32
    w = image.size[0] / scale
    w = int(w)
    image = image.resize((w, 32), Image.BILINEAR)
    image = np.array(image.convert('L')) / 255.0
    image = image.astype(np.float32)
    h, w = image.shape
    if w < 8:
        return ''

    image = torch.from_numpy(image)
    image = image.cuda() if GPU else image
    image = image.view(1, 1, *image.size())

    preds = model(image)
    preds = preds[:, 0, :]

    raw = transcription_dict(preds, alphabet)

    return raw


def label_encoding(json_dir):
    alphabet = process_alphabet(json_dir)
    encoding = [i for i in range(len(alphabet))]
    label_dict = {}
    for i in range(len(alphabet)):
        label_dict[alphabet[i]] = encoding[i]
    return label_dict


def get_batch_label(d, i):
    label = []
    for idx in i:
        label.append(list(d.labels[idx].values())[0])
    return label


def process_alphabet(json_dir):
    with open(json_dir, encoding='utf-8') as f:
        alphabet = json.loads(f.read())
    alphabet = alphabet + '丨 '
    return alphabet


class strLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, args, ignore_case=False):
        alphabet = process_alphabet(args.alphabet_dir)
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            # NOTE: n-1 is reserved for 'blank' if using pytorch_ctc
            self.dict[char] = i

    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """

        length = []
        result = []
        decode_flag = True if type(text[0])==bytes else False

        for item in text:

            if decode_flag:
                item = item.decode('utf-8','strict')
            length.append(len(item))
            for char in item:
                index = self.dict[char]
                result.append(index)
        text = result
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """

        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != len(self.alphabet)-1 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts

    def label_dict(self):

        return self.dict

    def label_constant(self):

        return self.alphabet



if __name__ == '__main__':

    sys.path.append('../')
    from model.cnn import CNN

    img = Image.open('../test/ocr/test.jpg')

    alphabet = process_alphabet('../weight/ocr/ocr.json')

    model = CNN(1).cuda()
    checkpoint = torch.load('../weight/ocr/ocr.pth.tar')
    model.load_state_dict(checkpoint)

    res = predict(model, img, alphabet)

    print(res)



