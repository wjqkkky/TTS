import numpy as np
import torch
from torch import nn
from torch.nn import functional
from TTS.tts.utils.generic_utils import sequence_mask


class L1LossMasked(nn.Module):

    def __init__(self, seq_len_norm):
        super(L1LossMasked, self).__init__()
        self.seq_len_norm = seq_len_norm

    def forward(self, x, target, length):
        """
        Args:
            x: A Variable containing a FloatTensor of size
                (batch, max_len, dim) which contains the
                unnormalized probability for each class.
            target: A Variable containing a LongTensor of size
                (batch, max_len, dim) which contains the index of the true
                class for each corresponding step.
            length: A Variable containing a LongTensor of size (batch,)
                which contains the length of each data in a batch.
        Returns:
            loss: An average loss value in range [0, 1] masked by the length.
        """
        # mask: (batch, max_len, 1)
        target.requires_grad = False
        mask = sequence_mask(
            sequence_length=length, max_len=target.size(1)).unsqueeze(2).float()
        if self.seq_len_norm:
            norm_w = mask / mask.sum(dim=1, keepdim=True)
            out_weights = norm_w.div(target.shape[0] * target.shape[2])
            mask = mask.expand_as(x)
            loss = functional.l1_loss(
                x * mask, target * mask, reduction='none')
            loss = loss.mul(out_weights.to(loss.device)).sum()
        else:
            mask = mask.expand_as(x)
            loss = functional.l1_loss(
                x * mask, target * mask, reduction='sum')
            loss = loss / mask.sum()
        return loss


class MSELossMasked(nn.Module):

    def __init__(self, seq_len_norm):
        super(MSELossMasked, self).__init__()
        self.seq_len_norm = seq_len_norm

    def forward(self, x, target, length):
        """
        Args:
            x: A Variable containing a FloatTensor of size
                (batch, max_len, dim) which contains the
                unnormalized probability for each class.
            target: A Variable containing a LongTensor of size
                (batch, max_len, dim) which contains the index of the true
                class for each corresponding step.
            length: A Variable containing a LongTensor of size (batch,)
                which contains the length of each data in a batch.
        Returns:
            loss: An average loss value in range [0, 1] masked by the length.
        """
        # mask: (batch, max_len, 1)
        target.requires_grad = False
        mask = sequence_mask(
            sequence_length=length, max_len=target.size(1)).unsqueeze(2).float()
        if self.seq_len_norm:
            norm_w = mask / mask.sum(dim=1, keepdim=True)
            out_weights = norm_w.div(target.shape[0] * target.shape[2])
            mask = mask.expand_as(x)
            loss = functional.mse_loss(
                x * mask, target * mask, reduction='none')
            loss = loss.mul(out_weights.to(loss.device)).sum()
        else:
            mask = mask.expand_as(x)
            loss = functional.mse_loss(
                x * mask, target * mask, reduction='sum')
            loss = loss / mask.sum()
        return loss


class AttentionEntropyLoss(nn.Module):
    # pylint: disable=R0201
    def forward(self, align):
        """
        Forces attention to be more decisive by penalizing
        soft attention weights

        TODO: arguments
        TODO: unit_test
        """
        entropy = torch.distributions.Categorical(probs=align).entropy()
        loss = (entropy / np.log(align.shape[1])).mean()
        return loss


class BCELossMasked(nn.Module):

    def __init__(self, pos_weight):
        super(BCELossMasked, self).__init__()
        self.pos_weight = pos_weight

    def forward(self, x, target, length):
        """
        Args:
            x: A Variable containing a FloatTensor of size
                (batch, max_len) which contains the
                unnormalized probability for each class.
            target: A Variable containing a LongTensor of size
                (batch, max_len) which contains the index of the true
                class for each corresponding step.
            length: A Variable containing a LongTensor of size (batch,)
                which contains the length of each data in a batch.
        Returns:
            loss: An average loss value in range [0, 1] masked by the length.
        """
        # mask: (batch, max_len, 1)
        target.requires_grad = False
        mask = sequence_mask(sequence_length=length, max_len=target.size(1)).float()
        loss = functional.binary_cross_entropy_with_logits(
            x * mask, target * mask, pos_weight=self.pos_weight, reduction='sum')
        loss = loss / mask.sum()
        return loss


class GuidedAttentionLoss(torch.nn.Module):
    def __init__(self, sigma=0.4):
        super(GuidedAttentionLoss, self).__init__()
        self.sigma = sigma

    def _make_ga_masks(self, ilens, olens):
        B = len(ilens)
        max_ilen = max(ilens)
        max_olen = max(olens)
        ga_masks = torch.zeros((B, max_olen, max_ilen))
        for idx, (ilen, olen) in enumerate(zip(ilens, olens)):
            ga_masks[idx, :olen, :ilen] = self._make_ga_mask(ilen, olen, self.sigma)
        return ga_masks

    def forward(self, att_ws, ilens, olens):
        ga_masks = self._make_ga_masks(ilens, olens).to(att_ws.device)
        seq_masks = self._make_masks(ilens, olens).to(att_ws.device)
        losses = ga_masks * att_ws
        loss = torch.mean(losses.masked_select(seq_masks))
        return loss

    @staticmethod
    def _make_ga_mask(ilen, olen, sigma):
        grid_x, grid_y = torch.meshgrid(torch.arange(olen), torch.arange(ilen))
        grid_x, grid_y = grid_x.float(), grid_y.float()
        return 1.0 - torch.exp(-(grid_y / ilen - grid_x / olen) ** 2 / (2 * (sigma ** 2)))

    @staticmethod
    def _make_masks(ilens, olens):
        in_masks = sequence_mask(ilens)
        out_masks = sequence_mask(olens)
        return out_masks.unsqueeze(-1) & in_masks.unsqueeze(-2)

class SpeakerEncoderLoss(torch.nn.Module):
    def __init__(self, c, beta=1):
        super(SpeakerEncoderLoss, self).__init__()
        self.beta = beta
        self.config = c
        #self.criterion = nn.MSELoss()
    def Lcycle(self, input_speaker_embeddings, output_speaker_embeddings):
        ''' 
        Reference: https://arxiv.org/pdf/1802.06984.pdf
        '''
        return torch.abs(input_speaker_embeddings-output_speaker_embeddings).pow(2).sum()

    def forward(self, speaker_encoder_model, mel_input, decoder_output, output_lens, loss_speaker_embeddings):
        # we can compute direct from input, but the zero padding is considered in loss, and its not is good
        # input_speaker_embeddings = self.speaker_encoder_model.inference(mel_input).detach()
        # output_speaker_embeddings = self.speaker_encoder_model.inference(decoder_output).detach()
        # the better choise is remove the padding and compute embedding but its more slow ...
        inp_se_list = []
        out_se_list = []
        for i in range(output_lens.size(0)):
            # remove the zero padding and compute speaker embedding
            if not self.config.use_data_aumentation_speakers:
                inp_emb = speaker_encoder_model.compute_embedding(mel_input[i][:output_lens[i], :].unsqueeze(0))
            out_emb = speaker_encoder_model.compute_embedding(decoder_output[i][:output_lens[i], :].unsqueeze(0))
            # append in a list
            if not self.config.use_data_aumentation_speakers:
                inp_se_list.append(inp_emb)
            out_se_list.append(out_emb)
        # convert lists to tensors and detach for not update weight in speaker encoder
        if not self.config.use_data_aumentation_speakers:
            input_speaker_embeddings = torch.stack(inp_se_list, dim=0).detach()
        else:
            input_speaker_embeddings = loss_speaker_embeddings
        output_speaker_embeddings = torch.stack(out_se_list, dim=0).detach()
        # compute and return loss
        return self.beta * self.Lcycle(input_speaker_embeddings, output_speaker_embeddings)

class TacotronLoss(torch.nn.Module):
    def __init__(self, c, stopnet_pos_weight=10, ga_sigma=0.4, speaker_encoder_model=None):
        super(TacotronLoss, self).__init__()
        self.stopnet_pos_weight = stopnet_pos_weight
        self.ga_alpha = c.ga_alpha
        self.config = c
        self.speaker_encoder_model = speaker_encoder_model
        self.num_real_samples = c.data_aumentation_num_real_samples
        # postnet decoder loss
        if c.loss_masking:
            self.criterion = L1LossMasked(c.seq_len_norm) if c.model in [
                "Tacotron"
            ] else MSELossMasked(c.seq_len_norm)
        else:
            self.criterion = nn.L1Loss() if c.model in ["Tacotron"
                                                        ] else nn.MSELoss()
        # guided attention loss
        if c.ga_alpha > 0:
            self.criterion_ga = GuidedAttentionLoss(sigma=ga_sigma)
        # stopnet loss
        # pylint: disable=not-callable
        self.criterion_st = BCELossMasked(pos_weight=torch.tensor(stopnet_pos_weight)) if c.stopnet else None

        if c.use_speaker_encoder_loss:
            self.criterion_se = SpeakerEncoderLoss(c=c, beta=c.speaker_encoder_loss_beta)

    def forward(self, postnet_output, decoder_output, mel_input, linear_input,
                stopnet_output, stopnet_target, output_lens, decoder_b_output,
                alignments, alignment_lens, alignments_backwards, input_lens,
                loss_speaker_embeddings=None):
        return_dict = {}
        # decoder and postnet losses
        if self.config.loss_masking:
            if self.config.use_data_aumentation_speakers:
                decoder_loss = self.criterion(decoder_output[:self.num_real_samples], mel_input[:self.num_real_samples],
                                            output_lens[:self.num_real_samples])
            else:
                decoder_loss = self.criterion(decoder_output, mel_input,
                                            output_lens)

            if self.config.model in ["Tacotron", "TacotronGST"]:
                if self.config.use_data_aumentation_speakers:
                    postnet_loss = self.criterion(postnet_output[:self.num_real_samples], linear_input[:self.num_real_samples],
                                                output_lens[:self.num_real_samples])
                else:
                    postnet_loss = self.criterion(postnet_output, linear_input,
                                                output_lens)
            else:
                if self.config.use_data_aumentation_speakers:
                    postnet_loss = self.criterion(postnet_output[:self.num_real_samples], mel_input[:self.num_real_samples],
                                                output_lens[:self.num_real_samples])
                else:
                    postnet_loss = self.criterion(postnet_output, mel_input,
                                                output_lens)
        else:
            if self.config.use_data_aumentation_speakers:
                decoder_loss = self.criterion(decoder_output[:self.num_real_samples], mel_input[:self.num_real_samples])
            else:
                decoder_loss = self.criterion(decoder_output, mel_input)
    
            if self.config.model in ["Tacotron", "TacotronGST"]:
                if self.config.use_data_aumentation_speakers:
                    postnet_loss = self.criterion(postnet_output[:self.num_real_samples], linear_input[:self.num_real_samples])
                else:
                    postnet_loss = self.criterion(postnet_output, linear_input)
            else:
                if self.config.use_data_aumentation_speakers:
                    postnet_loss = self.criterion(postnet_output[:self.num_real_samples], mel_input[:self.num_real_samples])
                else:
                    postnet_loss = self.criterion(postnet_output, mel_input)

        loss = decoder_loss + postnet_loss
        return_dict['decoder_loss'] = decoder_loss
        return_dict['postnet_loss'] = postnet_loss

        # stopnet loss
        stop_loss = self.criterion_st(
            stopnet_output, stopnet_target,
            output_lens) if self.config.stopnet else torch.zeros(1)
        if not self.config.separate_stopnet and self.config.stopnet:
            loss += stop_loss
        return_dict['stopnet_loss'] = stop_loss

        # backward decoder loss (if enabled)
        if self.config.bidirectional_decoder:
            if self.config.loss_masking:
                if self.config.use_data_aumentation_speakers:
                    decoder_b_loss = self.criterion(torch.flip(decoder_b_output[:self.num_real_samples], dims=(1, )), mel_input[:self.num_real_samples], output_lens[:self.num_real_samples])
                else:
                    decoder_b_loss = self.criterion(torch.flip(decoder_b_output, dims=(1, )), mel_input, output_lens)
            else:
                if self.config.use_data_aumentation_speakers:
                    decoder_b_loss = self.criterion(torch.flip(decoder_b_output[:self.num_real_samples], dims=(1, )), mel_input[:self.num_real_samples])
                else:
                    decoder_b_loss = self.criterion(torch.flip(decoder_b_output, dims=(1, )), mel_input)
            # this loss we can compute if use use_data_aumentation_speakers or not because is independent to mel_input
            decoder_c_loss = torch.nn.functional.l1_loss(torch.flip(decoder_b_output, dims=(1, )), decoder_output)
            loss += decoder_b_loss + decoder_c_loss
            return_dict['decoder_b_loss'] = decoder_b_loss
            return_dict['decoder_c_loss'] = decoder_c_loss

        # double decoder consistency loss (if enabled)
        if self.config.double_decoder_consistency:
            if self.config.use_data_aumentation_speakers:
                decoder_b_loss = self.criterion(decoder_b_output[:self.num_real_samples], mel_input[:self.num_real_samples], output_lens[:self.num_real_samples])
            else:
                decoder_b_loss = self.criterion(decoder_b_output, mel_input, output_lens)
            # decoder_c_loss = torch.nn.functional.l1_loss(decoder_b_output, decoder_output)
            attention_c_loss = torch.nn.functional.l1_loss(alignments, alignments_backwards)
            loss += decoder_b_loss + attention_c_loss
            return_dict['decoder_coarse_loss'] = decoder_b_loss
            return_dict['decoder_ddc_loss'] = attention_c_loss

        # guided attention loss (if enabled)
        if self.config.ga_alpha > 0:
            ga_loss = self.criterion_ga(alignments, input_lens, alignment_lens)
            loss += ga_loss * self.ga_alpha
            return_dict['ga_loss'] = ga_loss * self.ga_alpha

        # speaker encoder extra loss (if enabled)
        if self.config.use_speaker_encoder_loss:
            se_loss = self.criterion_se(self.speaker_encoder_model, mel_input, decoder_output, output_lens, loss_speaker_embeddings)
            loss += se_loss
            return_dict['se_loss'] = se_loss

        return_dict['loss'] = loss
        return return_dict

