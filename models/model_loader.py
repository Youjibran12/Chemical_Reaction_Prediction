import os
import torch
from huggingface_hub import hf_hub_download
from tokenizer.smiles_tokenizer import SMILESTokenizer
from models.reaction_predictor import ReactionPredictor
from models.gru_seq2seq import GRUSeq2Seq
from models.transformer_model import TransformerModel
from models.ensemble_model import EnsemblePredictor

def save_models(gru_predictor, tf_predictor, save_dir='saved_models'):
    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        'model_state_dict': gru_predictor.model.state_dict(),
        'vocab': gru_predictor.tokenizer.vocab,
        'stoi':  gru_predictor.tokenizer.stoi,
        'itos':  gru_predictor.tokenizer.itos,
        'model_type': 'gru',
        'max_len': gru_predictor.max_len,
    }, os.path.join(save_dir, 'gru_model.pt'))
    print(f'GRU saved  → {save_dir}/gru_model.pt')

    torch.save({
        'model_state_dict': tf_predictor.model.state_dict(),
        'vocab': tf_predictor.tokenizer.vocab,
        'stoi':  tf_predictor.tokenizer.stoi,
        'itos':  tf_predictor.tokenizer.itos,
        'model_type': 'transformer',
        'max_len': tf_predictor.max_len,
    }, os.path.join(save_dir, 'transformer_model.pt'))
    print(f'Transformer saved → {save_dir}/transformer_model.pt')

from huggingface_hub import hf_hub_download

def load_ensemble_from_disk(gru_path, tf_path, device=None,
                             gru_weight=0.4, transformer_weight=0.6):

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    

    gru_path = hf_hub_download(
        repo_id="saketjaju24/chemical-reaction-model",
        filename="gru_model.pt"
    )

    tf_path = hf_hub_download(
        repo_id="saketjaju24/chemical-reaction-model",
        filename="transformer_model.pt"
    )
    gru_ckpt = torch.load(gru_path,  map_location=device, weights_only=True)
    tf_ckpt  = torch.load(tf_path,   map_location=device, weights_only=True)

    # Vocab alignment guard
    assert gru_ckpt['vocab'] == tf_ckpt['vocab'], (
        'Vocab mismatch! GRU and Transformer were saved with different '
        'tokenizers. Retrain both from the same vocab.'
    )
    assert gru_ckpt['stoi'] == tf_ckpt['stoi'], (
        'Token-to-ID mapping mismatch! Models cannot be safely ensembled.'
    )
    print('Vocab check passed — both models share identical tokenizers')

    # Rebuild shared tokenizer
    tokenizer       = SMILESTokenizer()
    tokenizer.vocab = gru_ckpt['vocab']
    tokenizer.stoi  = gru_ckpt['stoi']
    tokenizer.itos  = gru_ckpt['itos']
    vocab_size      = len(tokenizer.vocab)

    # Rebuild GRU
    gru_pred           = ReactionPredictor(model_type='gru')
    gru_pred.tokenizer = tokenizer
    gru_pred.device    = device
    gru_pred.model     = GRUSeq2Seq(vocab_size).to(device)
    gru_pred.model.load_state_dict(gru_ckpt['model_state_dict'])

    # Rebuild Transformer
    tf_pred           = ReactionPredictor(model_type='transformer')
    tf_pred.tokenizer = tokenizer
    tf_pred.device    = device
    tf_pred.model     = TransformerModel(vocab_size).to(device)
    tf_pred.model.load_state_dict(tf_ckpt['model_state_dict'])

    return EnsemblePredictor(gru_pred, tf_pred,
                              gru_weight=gru_weight,
                              transformer_weight=transformer_weight)


ensemble = load_ensemble_from_disk(
    'gru_model.pt',
    'transformer_model.pt',
    gru_weight=0.7,
    transformer_weight=0.3
)

gru_predictor = ensemble.gru
tf_predictor  = ensemble.tf