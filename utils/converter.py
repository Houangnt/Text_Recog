import numpy as np
import torch
from torch.autograd import Variable


class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character, device):
        # character (str): set of the possible characters.
        dict_character = list(character)
        self.device = device

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = ['[CTCblank]'] + dict_character  # dummy '[CTCblank]' token for CTCLoss (index 0)

    def encode(self, text, batch_max_length=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text: text index for CTCLoss. [batch_size, batch_max_length]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]

        # The index used for padding (=0) would not affect the CTC loss calculation.
        batch_text = torch.LongTensor(len(text), batch_max_length).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text = [self.dict[char] for char in text]
            batch_text[i][:len(text)] = torch.LongTensor(text)
        return batch_text.to(self.device), torch.IntTensor(length).to(self.device)

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            t = text_index[index, :]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
        return texts


class CTCLabelConverterForBaiduWarpctc(object):
    """ Convert between text-label and text-index for baidu warpctc """

    def __init__(self, character):
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = ['[CTCblank]'] + dict_character  # dummy '[CTCblank]' token for CTCLoss (index 0)

    def encode(self, text, batch_max_length=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]
        text = ''.join(text)
        text = [self.dict[char] for char in text]

        return torch.IntTensor(text), torch.IntTensor(length)

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        index = 0
        for l in length:
            t = text_index[index:index + l]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
            index += l
        return texts


class AttnLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character, device):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character
        self.device = device

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i

    def encode(self, text, batch_max_length=25):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        return (batch_text.to(self.device), torch.IntTensor(length).to(self.device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts


class TransformerConverter(object):
    """
    class to decode a word during evaluation
    """

    def __init__(self, character, device=torch.device('cuda')):
        """
        :param character (str): set of the possible characters.
        :param device: GPU or CPU
        """
        super(TransformerConverter, self).__init__()
        self.ended = False
        self.device = device
        self.character = ['[GO]', '[PAD]', '[EOS]'] + list(character)

        self.dict = {}
        for i, char in enumerate(self.character):
            self.dict[char] = i

    def greedy_decode(self, out, targets, characters, probability, top_k=1):
        """

        :param out: output distribution over characters
        :param targets: target mask
        :param characters: current characters
        :param probability: current probability
        :param top_k: top k predictions
        :return: predicted character, character_id, and probability
        """
        ended = False

        if out is None:
            ended = True
            probability = -1

        else:
            prob, char_id = torch.topk(out[:, -1], top_k)
            next_char = self.character[char_id.data[0]]
            probability *= torch.exp(prob).tolist()[0][0]
            targets = torch.cat((targets, torch.tensor([[char_id]]).to(self.device)), 1)
            if char_id == 2:  # EOS ID
                ended = True
            else:
                characters += next_char

        return characters, probability, targets, ended

    def encode(self, texts, batch_max_length=25):
        ltr_batch = None
        rtl_batch = None
        target_y_batch = None
        mask_batch = None
        targets_embedding_mask_batch = None
        for text in texts:
            token_ids = [self.dict[char] for char in text.rstrip()]
            reversed_tokes = list(reversed(token_ids))
            ltr = torch.tensor([0] + token_ids + [2] + [1] * (batch_max_length - len(text.rstrip())), dtype=torch.int32)
            rtl = torch.tensor([0] + reversed_tokes + [2] + [1] * (batch_max_length - len(text.rstrip())),
                               dtype=torch.int32)
            ltr_y = self.one_hot_targets(ltr, batch_max_length)[1:]
            rtl_y = self.one_hot_targets(rtl, batch_max_length)[1:]
            target_y = torch.from_numpy(np.concatenate((ltr_y, rtl_y), 0))
            mask = self.make_mask(ltr, batch_max_length)
            ltr = ltr[:-1]
            rtl = rtl[:-1]
            targets_embedding_mask = self._make_std_mask(ltr, 1).to(self.device)

            mask_batch = mask.unsqueeze(0) if mask_batch is None else torch.cat((mask_batch, mask.unsqueeze(0)))
            ltr_batch = ltr.unsqueeze(0) if ltr_batch is None else torch.cat((ltr_batch, ltr.unsqueeze(0)))
            rtl_batch = rtl.unsqueeze(0) if rtl_batch is None else torch.cat((rtl_batch, rtl.unsqueeze(0)))
            target_y_batch = target_y.unsqueeze(0) if target_y_batch is None \
                else torch.cat((target_y_batch, target_y.unsqueeze(0)))
            targets_embedding_mask_batch = targets_embedding_mask.unsqueeze(
                0) if targets_embedding_mask_batch is None else torch.cat((targets_embedding_mask_batch,
                                                                           targets_embedding_mask.unsqueeze(0)))

        return (ltr_batch.to(self.device), rtl_batch.to(self.device), target_y_batch.to(self.device),
                mask_batch.to(self.device), targets_embedding_mask_batch.to(self.device)), []

    def one_hot_targets(self, y, batch_max_length):
        """
        convert characters indexes in the y target into one hot vector
        :param y: target vector, shape [1, max_sequence length] with the indexes of the characters
        :param batch_max_length: maximum length the characters
        :return: one hot representation [max_sequence, len(self.CHARMAP)]
        """
        try:
            one_hot = np.zeros((batch_max_length + 2, len(self.dict)))
            one_hot[np.arange(batch_max_length + 2), y] = 1
        except Exception as e:
            print(y)
            raise e

        return one_hot

    def make_mask(self, target, batch_max_length):
        """
        make a mask to mask all the padding symbols for the loss calculation
        :param target:
        :param batch_max_length: maximum length the characters
        :return: numpy array with mask
        """

        mask = np.ones((batch_max_length + 2, len(self.dict)))
        mask[np.where(target == 1), :] = 0
        return torch.from_numpy(mask[:-1, :])

    def _make_std_mask(self, target, pad_id):
        """
        Create a mask to hide padding and future words
        :param target: target tensor
        :param pad_id: id of the padding symbol
        :return: target mask, to mask all the predictions after the EOW_ID
        """

        tgt_mask = (target != pad_id).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(self.subsequent_mask(target.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

    def get_init_target(self):
        return torch.tensor([[0]]).to(self.device)


    @staticmethod
    def subsequent_mask(size):
        """
        Mask out subsequent positions.
        :param size:
        :return:
        """

        attn_shape = (1, size, size)
        subsequent_msk = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        return torch.from_numpy(subsequent_msk) == 0


def attn(opt):
    return AttnLabelConverter(opt.character, opt.device)


def ctc(opt):
    return CTCLabelConverter(opt.character, opt.device)


def transformer(opt):
    return TransformerConverter(opt.character, opt.device)
