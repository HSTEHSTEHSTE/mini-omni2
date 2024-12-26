import os
import lightning as L
import torch
import glob
import time
from snac import SNAC
from litgpt import Tokenizer
from litgpt.utils import (
    num_parameters,
)
from litgpt.generate.base import (
    generate_TA,
    generate_AT,
    next_token_image_batch
)
import soundfile as sf
from litgpt.model import GPT, Config
from lightning.fabric.utilities.load import _lazy_load as lazy_load
from utils.snac_utils import layershift, reconscruct_snac, reconstruct_tensors, get_time_str
from utils.snac_utils import get_snac, generate_audio_data
import whisper
from tqdm import tqdm
from huggingface_hub import snapshot_download


torch.set_printoptions(sci_mode=False)


# TODO
text_vocabsize = 151936
text_specialtokens = 64
audio_vocabsize = 4096
audio_specialtokens = 64

padded_text_vocabsize = text_vocabsize + text_specialtokens
padded_audio_vocabsize = audio_vocabsize + audio_specialtokens

_eot = text_vocabsize
_pad_t = text_vocabsize + 1
_input_t = text_vocabsize + 2
_answer_t = text_vocabsize + 3
_asr = text_vocabsize + 4

_eoa = audio_vocabsize
_pad_a = audio_vocabsize + 1
_input_a = audio_vocabsize + 2
_answer_a = audio_vocabsize + 3
_split = audio_vocabsize + 4
_image = audio_vocabsize + 5
_eoimage = audio_vocabsize + 6


def get_input_ids_TT(text, text_tokenizer):
    input_ids_item = [[] for i in range(8)]
    text_tokens = text_tokenizer.encode(text).tolist()

    for i in range(7):
        input_ids_item[i] = torch.tensor(
            [layershift(_pad_a, i)] * (len(text_tokens) + 3)
        ).unsqueeze(0)
    input_ids_item[-1] = [_input_t] + text_tokens + [_eot] + [_answer_t]
    input_ids_item[-1] = torch.tensor(input_ids_item[-1]).unsqueeze(0)

    return input_ids_item


def get_input_ids_whisper(
    mel, leng, whispermodel, device, 
    special_token_a=_answer_a, special_token_t=_answer_t,
):

    with torch.no_grad():
        mel = mel.unsqueeze(0).to(device)
        # audio_feature = whisper.decode(whispermodel,mel, options).audio_features
        audio_feature = whispermodel.embed_audio(mel)[0][:leng]

    T = audio_feature.size(0)
    input_ids = []
    for i in range(7):
        input_ids_item = []
        input_ids_item.append(layershift(_input_a, i))
        input_ids_item += [layershift(_pad_a, i)] * T
        input_ids_item += [(layershift(_eoa, i)), layershift(special_token_a, i)]
        input_ids.append(torch.tensor(input_ids_item).unsqueeze(0))
    input_id_T = torch.tensor([_input_t] + [_pad_t] * T + [_eot, special_token_t])
    input_ids.append(input_id_T.unsqueeze(0))
    return audio_feature.unsqueeze(0), input_ids


def get_input_ids_whisper_ATBatch(mel, leng, whispermodel, device):
    with torch.no_grad():
        mel = mel.unsqueeze(0).to(device)
        # audio_feature = whisper.decode(whispermodel,mel, options).audio_features
        audio_feature = whispermodel.embed_audio(mel)[0][:leng]
    T = audio_feature.size(0)
    input_ids_AA = []
    for i in range(7):
        input_ids_item = []
        input_ids_item.append(layershift(_input_a, i))
        input_ids_item += [layershift(_pad_a, i)] * T
        input_ids_item += [(layershift(_eoa, i)), layershift(_answer_a, i)]
        input_ids_AA.append(torch.tensor(input_ids_item))
    input_id_T = torch.tensor([_input_t] + [_pad_t] * T + [_eot, _answer_t])
    input_ids_AA.append(input_id_T)

    input_ids_AT = []
    for i in range(7):
        input_ids_item = []
        input_ids_item.append(layershift(_input_a, i))
        input_ids_item += [layershift(_pad_a, i)] * T
        input_ids_item += [(layershift(_eoa, i)), layershift(_pad_a, i)]
        input_ids_AT.append(torch.tensor(input_ids_item))
    input_id_T = torch.tensor([_input_t] + [_pad_t] * T + [_eot, _answer_t])
    input_ids_AT.append(input_id_T)

    input_ids = [input_ids_AA, input_ids_AT]
    stacked_inputids = [[] for _ in range(8)]
    for i in range(2):
        for j in range(8):
            stacked_inputids[j].append(input_ids[i][j])
    stacked_inputids = [torch.stack(tensors) for tensors in stacked_inputids]
    return torch.stack([audio_feature, audio_feature]), stacked_inputids


def load_audio(path):
    audio = whisper.load_audio(path)
    duration_ms = (len(audio) / 16000) * 1000
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio)
    return mel, int(duration_ms / 20) + 1


def A1_T2(fabric, audio_feature, input_ids, leng, model, text_tokenizer, step, audio_suffix=None):
    with fabric.init_tensor():
        model.set_kv_cache(batch_size=1)
    tokenlist = generate_AT(
        model,
        audio_feature,
        input_ids,
        [leng],
        ["AT"],
        max_returned_tokens=2048,
        temperature=0.9,
        top_k=1,
        eos_id_a=_eoa,
        eos_id_t=_eot,
        pad_id_t=_pad_t,
        shift=padded_text_vocabsize,
        include_prompt=True,
        generate_text=True,
        audio_suffix=audio_suffix,
    )
    return text_tokenizer.decode(torch.tensor(tokenlist)).strip()

    
def load_model(ckpt_dir, device):
    snacmodel = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)
    whisper_model_path = ckpt_dir + "/small.pt"
    if not os.path.exists(whisper_model_path):
        whisper_model_path = "small"
    whispermodel = whisper.load_model(whisper_model_path).to(device)
    text_tokenizer = Tokenizer(ckpt_dir)
    fabric = L.Fabric(devices=1, strategy="auto")
    config = Config.from_file(ckpt_dir + "/model_config.yaml")
    config.post_adapter = False

    with fabric.init_module(empty_init=False):
        model = GPT(config)

    model = fabric.setup(model)
    state_dict = lazy_load(ckpt_dir + "/lit_model.pth")
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    return fabric, model, text_tokenizer, snacmodel, whispermodel

    
def download_model(ckpt_dir):
    repo_id = "gpt-omni/mini-omni2"
    snapshot_download(repo_id, local_dir=ckpt_dir, revision="main")

    
def get_text_stream(list_output, index, text_tokenizer):
    text_tokens = list_output[-1][index:]
    index += len(text_tokens)
    is_text_end = False
    if text_vocabsize in text_tokens:
        text_tokens = text_tokens[:text_tokens.index(text_vocabsize)]
        is_text_end = True
    if len(text_tokens) == 0:
        return "", index, is_text_end
    res_text = text_tokenizer.decode(torch.tensor(text_tokens))
    return res_text, index, is_text_end

    
class OmniInference:

    def __init__(self, ckpt_dir='./checkpoint', device='cuda:0'):
        self.device = device
        if not os.path.exists(ckpt_dir):
            print(f"checkpoint directory {ckpt_dir} not found, downloading from huggingface")
            download_model(ckpt_dir)
        self.fabric, self.model, self.text_tokenizer, self.snacmodel, self.whispermodel = load_model(ckpt_dir, device)

    def warm_up(self, sample='./data/samples/output1.wav'):
        for _ in self.run_AT_batch_stream(sample):
            pass

    @torch.inference_mode()
    def run_AT_batch_stream(self, 
                            audio_path, 
                            stream_stride=4,
                            max_returned_tokens=2048, 
                            temperature=0.9, 
                            top_k=1, 
                            top_p=1.0,
                            eos_id_a=_eoa,
                            eos_id_t=_eot,
                            save_path=None
        ):

        assert os.path.exists(audio_path), f"audio file {audio_path} not found"
        model = self.model

        with self.fabric.init_tensor():
            model.set_kv_cache(batch_size=2,device=self.device)

        mel, leng = load_audio(audio_path)
        audio_feature, input_ids = get_input_ids_whisper_ATBatch(mel, leng, self.whispermodel, self.device)
        T = input_ids[0].size(1)
        device = input_ids[0].device

        assert max_returned_tokens > T, f"max_returned_tokens {max_returned_tokens} should be greater than audio length {T}"

        if model.max_seq_length < max_returned_tokens - 1:
            raise NotImplementedError(
                f"max_seq_length {model.max_seq_length} needs to be >= {max_returned_tokens - 1}"
            )

        input_pos = torch.tensor([T], device=device)
        list_output = [[] for i in range(8)]
        tokens_A, token_T = next_token_image_batch(
            model,
            audio_feature.to(torch.float32).to(model.device),
            None,
            input_ids,
            [T - 3, T - 3],
            ["A1T2", "A1T2"],
            input_pos=torch.arange(0, T, device=device),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        for i in range(7):
            list_output[i].append(tokens_A[i].tolist()[0])
        list_output[7].append(token_T.tolist()[0])

        model_input_ids = [[] for i in range(8)]
        for i in range(7):
            tokens_A[i] = tokens_A[i].clone() + padded_text_vocabsize + i * padded_audio_vocabsize
            model_input_ids[i].append(tokens_A[i].clone().to(device).to(torch.int32))
            model_input_ids[i].append(torch.tensor([layershift(4097, i)], device=device))
            model_input_ids[i] = torch.stack(model_input_ids[i])

        model_input_ids[-1].append(token_T.clone().to(torch.int32))
        model_input_ids[-1].append(token_T.clone().to(torch.int32))
        model_input_ids[-1] = torch.stack(model_input_ids[-1])

        text_end = False
        index = 1
        nums_generate = stream_stride
        begin_generate = False
        current_index = 0

        text_index = 0
        is_text_end = False

        for _ in tqdm(range(2, max_returned_tokens - T + 1)):
            tokens_A, token_T = next_token_image_batch(
                model,
                None,
                None,
                model_input_ids,
                None,
                None,
                input_pos=input_pos,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            if text_end:
                token_T = torch.tensor([_pad_t], device=device)

            if tokens_A[-1] == eos_id_a:
                break

            if token_T == eos_id_t:
                text_end = True

            for i in range(7):
                list_output[i].append(tokens_A[i].tolist()[0])
            list_output[7].append(token_T.tolist()[0])

            model_input_ids = [[] for i in range(8)]
            for i in range(7):
                tokens_A[i] = tokens_A[i].clone() +padded_text_vocabsize + i * padded_audio_vocabsize
                model_input_ids[i].append(tokens_A[i].clone().to(device).to(torch.int32))
                model_input_ids[i].append(
                    torch.tensor([layershift(4097, i)], device=device)
                )
                model_input_ids[i] = torch.stack(model_input_ids[i])

            model_input_ids[-1].append(token_T.clone().to(torch.int32))
            model_input_ids[-1].append(token_T.clone().to(torch.int32))
            model_input_ids[-1] = torch.stack(model_input_ids[-1])

            if index == 7:
                begin_generate = True

            if begin_generate:
                current_index += 1
                if current_index == nums_generate:
                    current_index = 0
                    snac = get_snac(list_output, index, nums_generate)
                    audio_stream = generate_audio_data(snac, self.snacmodel, self.device)
                    if is_text_end:
                        text_stream = ""
                    else:
                        text_stream, text_index, is_text_end = get_text_stream(list_output, text_index, self.text_tokenizer)

                    yield (audio_stream, text_stream)

            input_pos = input_pos.add_(1)
            index += 1
        text = self.text_tokenizer.decode(torch.tensor(list_output[-1]))
        print(f"text output: {text}")

        if save_path is not None:
            audiolist = reconscruct_snac(list_output)
            audio = reconstruct_tensors(audiolist)
            with torch.inference_mode():
                audio_hat = self.snacmodel.decode(audio)
                sf.write(save_path, audio_hat.squeeze().cpu().numpy(), 24000)
        
        model.clear_kv_cache()
        return list_output


def test_infer():
    device = "cuda:0"
    out_dir = f"./output/{get_time_str()}"
    ckpt_dir = f"./checkpoint"
    if not os.path.exists(ckpt_dir):
        print(f"checkpoint directory {ckpt_dir} not found, downloading from huggingface")
        download_model(ckpt_dir)

    fabric, model, text_tokenizer, snacmodel, whispermodel = load_model(ckpt_dir, device)

    task = ["AT"]

    # prepare test data
    # TODO
    test_audio_list = sorted(glob.glob('./data/samples/output*.wav'))

    # LOAD MODEL
    with torch.no_grad():
        if "AT" in task:
            print("===============================================================")
            print("                       testing A1T2")
            print("===============================================================")
            step = 0
            for path in test_audio_list:
                mel, leng = load_audio(path)
                audio_feature, input_ids = get_input_ids_whisper(
                    mel, leng, whispermodel, device, 
                    special_token_a=_pad_a, special_token_t=_answer_t
                )
                audio_suffix = "What is the language of the previous utterance?"
                audio_suffix_ids = get_input_ids_TT(audio_suffix, text_tokenizer)
                text = A1_T2(
                    fabric, audio_feature, input_ids, leng, model, text_tokenizer, step, audio_suffix=audio_suffix_ids
                )
                print(f"output: {text}")
                step += 1
                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("===============================================================")    

if __name__ == "__main__":
    test_infer()
