import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm

from tokenizer.smiles_tokenizer import SMILESTokenizer
from utils.dataset import ReactionDataset
from models.transformer_model import TransformerModel
from models.gru_seq2seq import GRUSeq2Seq
from evaluation.metrics import compute_metrics, print_metrics

class ReactionPredictor:
    def __init__(self, model_type='transformer', max_len=128):
        assert model_type in ('transformer', 'gru')
        self.device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer  = SMILESTokenizer()
        self.model      = None
        self.max_len    = max_len
        self.model_type = model_type

    def load_data(self, num_samples=100_000):
        dataset   = load_dataset('liupf/SLM4CRP_with_RTs', split='train')
        reactions = [
            r for r in dataset
            if len(r['input']) <= self.max_len and len(r['output']) <= self.max_len
        ]
        return reactions[:num_samples]

    def train(self, reactions, epochs=20, batch_size=64,
              gru_embed_dim=256, gru_hidden_dim=256, gru_num_layers=2,
              gru_dropout=0.1, gru_attn_dim=128):

        self.tokenizer.build_vocab(reactions)
        dataset    = ReactionDataset(reactions, self.tokenizer)
        train_size = int(0.8 * len(dataset))
        train_set, val_set = torch.utils.data.random_split(
            dataset, [train_size, len(dataset) - train_size]
        )
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                  num_workers=4, pin_memory=True, persistent_workers=True)
        val_loader   = DataLoader(val_set, batch_size=batch_size,
                                  num_workers=4, pin_memory=True, persistent_workers=True)

        vocab_size = len(self.tokenizer.vocab)

        if self.model_type == 'transformer':
            self.model = TransformerModel(vocab_size, d_model=512, nhead=8,
                                          num_layers=4, dropout=0.1).to(self.device)
            print('[Transformer] model created  (d=512, layers=4, ffn=2048)')
        else:
            self.model = GRUSeq2Seq(
                vocab_size,
                embed_dim=gru_embed_dim, hidden_dim=gru_hidden_dim,
                num_layers=gru_num_layers, dropout=gru_dropout, attn_dim=gru_attn_dim,
            ).to(self.device)
            print('[GRU Seq2Seq] model created')

        print('  torch.compile() skipped (keeps Ensemble compatibility)')

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'  Trainable parameters: {n_params:,}')

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=3e-4,
            steps_per_epoch=len(train_loader), epochs=epochs,
            pct_start=0.05, anneal_strategy='cos',
        )
        criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
        scaler    = torch.amp.GradScaler('cuda')

        best_val_loss = float('inf')
        train_losses, val_losses = [], []
        epoch_metrics = []
        ckpt_name = f'best_{self.model_type}.pt'
        _val_reactions = [reactions[i] for i in range(min(200, len(reactions)))]

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for src, tgt in train_loader:
                src = src.to(self.device, non_blocking=True)
                tgt = tgt.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    if self.model_type == 'transformer':
                        output = self.model(src, tgt[:, :-1])
                    else:
                        output = self.model(src, tgt)
                    loss = criterion(output.reshape(-1, output.size(-1)),
                                     tgt[:, 1:].reshape(-1))
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                train_loss += loss.item()

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad(), torch.amp.autocast('cuda'):
                for src, tgt in val_loader:
                    src = src.to(self.device, non_blocking=True)
                    tgt = tgt.to(self.device, non_blocking=True)
                    if self.model_type == 'transformer':
                        output = self.model(src, tgt[:, :-1])
                    else:
                        output = self.model(src, tgt)
                    val_loss += criterion(output.reshape(-1, output.size(-1)),
                                          tgt[:, 1:].reshape(-1)).item()

            avg_train = train_loss / len(train_loader)
            avg_val   = val_loss   / len(val_loader)
            train_losses.append(avg_train)
            val_losses.append(avg_val)
            current_lr = optimizer.param_groups[0]['lr']

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save(self.model.state_dict(), ckpt_name)

            _preds   = [self.predict(_r['input'], beam_width=1) for _r in _val_reactions[:50]]
            _actuals = [_r['output'] for _r in _val_reactions[:50]]
            _m = compute_metrics(_preds, _actuals)
            epoch_metrics.append({
                'epoch':         epoch + 1,
                'f1':            _m['f1'],
                'char_accuracy': _m['char_accuracy'],
                'exact_match':   _m['exact_match'],
            })
            print(f'Epoch {epoch+1:02d}/{epochs} | '
                  f'train: {avg_train:.4f} | val: {avg_val:.4f} | '
                  f'f1: {_m["f1"]:.3f} | char_acc: {_m["char_accuracy"]:.3f} | '
                  f'lr: {current_lr:.2e}')

        self.model.load_state_dict(torch.load(ckpt_name, weights_only=True))
        print(f'Best {self.model_type} model restored (val_loss={best_val_loss:.4f})')
        return train_losses, val_losses, epoch_metrics

    def predict(self, src_text, beam_width=5):
        self.model.eval()
        src   = torch.tensor([self.tokenizer.encode(src_text)]).to(self.device)
        START = self.tokenizer.stoi['[START]']
        END   = self.tokenizer.stoi['[END]']
        PAD   = self.tokenizer.stoi['[PAD]']

        with torch.no_grad():
            if self.model_type == 'transformer':
                beams     = [(0.0, [], torch.tensor([[START]]).to(self.device))]
                completed = []
                for _ in range(self.max_len):
                    new_beams = []
                    for score, tokens, tgt in beams:
                        logits    = self.model(src, tgt)[0, -1, :]
                        log_probs = torch.log_softmax(logits, dim=-1)
                        topk      = log_probs.topk(beam_width)
                        for lp, tok_id in zip(topk.values, topk.indices):
                            tok_id  = tok_id.item()
                            new_tok = tokens + [tok_id]
                            new_tgt = torch.cat(
                                [tgt, torch.tensor([[tok_id]]).to(self.device)], dim=1
                            )
                            if tok_id == END:
                                completed.append((score + lp.item(), new_tok[:-1]))
                            else:
                                new_beams.append((score + lp.item(), new_tok, new_tgt))
                    if not new_beams: break
                    new_beams.sort(key=lambda x: x[0] / max(len(x[1]), 1), reverse=True)
                    beams = new_beams[:beam_width]
                if completed:
                    completed.sort(key=lambda x: x[0] / max(len(x[1]), 1), reverse=True)
                    best_tokens = completed[0][1]
                else:
                    best_tokens = beams[0][1] if beams else []
                best_tokens = [t for t in best_tokens if t not in (END, PAD)]
                return self.tokenizer.decode(best_tokens)

            else:  # GRU
                src_mask = (src == PAD)
                encoder_outputs, hidden = self.model.encode(src)
                beams     = [(0.0, [], hidden)]
                completed = []
                for _ in range(self.max_len):
                    new_beams = []
                    for score, tokens, h in beams:
                        last_tok  = tokens[-1] if tokens else START
                        dec_input = torch.tensor([last_tok]).to(self.device)
                        logits, new_h, _ = self.model.decode_step(
                            dec_input, h, encoder_outputs, src_mask
                        )
                        log_probs = torch.log_softmax(logits[0], dim=-1)
                        topk      = log_probs.topk(beam_width)
                        for lp, tok_id in zip(topk.values, topk.indices):
                            tok_id  = tok_id.item()
                            new_tok = tokens + [tok_id]
                            if tok_id == END:
                                completed.append((score + lp.item(), new_tok[:-1]))
                            else:
                                new_beams.append((score + lp.item(), new_tok, new_h))
                    if not new_beams: break
                    new_beams.sort(key=lambda x: x[0] / max(len(x[1]), 1), reverse=True)
                    beams = new_beams[:beam_width]
                if completed:
                    completed.sort(key=lambda x: x[0] / max(len(x[1]), 1), reverse=True)
                    best_tokens = completed[0][1]
                else:
                    best_tokens = beams[0][1] if beams else []
                best_tokens = [t for t in best_tokens if t not in (END, PAD)]
                return self.tokenizer.decode(best_tokens)

    def predict_topk(self, src_text, beam_width=5):
        self.model.eval()
        src   = torch.tensor([self.tokenizer.encode(src_text)]).to(self.device)
        START = self.tokenizer.stoi['[START]']
        END   = self.tokenizer.stoi['[END]']
        PAD   = self.tokenizer.stoi['[PAD]']

        with torch.no_grad():
            if self.model_type == 'transformer':
                beams     = [(0.0, [START], torch.tensor([[START]]).to(self.device))]
                completed = []
                for _ in range(self.max_len):
                    new_beams = []
                    for score, tokens, tgt in beams:
                        logits    = self.model(src, tgt)[0, -1, :]
                        log_probs = torch.log_softmax(logits, dim=-1)
                        topk      = log_probs.topk(beam_width)
                        for lp, tok_id in zip(topk.values, topk.indices):
                            tok_id  = tok_id.item()
                            new_tok = tokens + [tok_id]
                            new_tgt = torch.cat(
                                [tgt, torch.tensor([[tok_id]]).to(self.device)], dim=1
                            )
                            if tok_id == END:
                                completed.append((score + lp.item(), new_tok[1:-1]))
                            else:
                                new_beams.append((score + lp.item(), new_tok, new_tgt))
                    if not new_beams: break
                    new_beams.sort(key=lambda x: x[0] / max(len(x[1]), 1), reverse=True)
                    beams = new_beams[:beam_width]
                for score, tokens, _ in beams:
                    completed.append((score, tokens[1:]))

            else:  # GRU
                src_mask = (src == PAD)
                enc_out, hidden = self.model.encode(src)
                beams     = [(0.0, [], hidden)]
                completed = []
                for _ in range(self.max_len):
                    new_beams = []
                    for score, tokens, h in beams:
                        last_tok  = tokens[-1] if tokens else START
                        dec_input = torch.tensor([last_tok]).to(self.device)
                        logits, new_h, _ = self.model.decode_step(
                            dec_input, h, enc_out, src_mask
                        )
                        log_probs = torch.log_softmax(logits[0], dim=-1)
                        topk      = log_probs.topk(beam_width)
                        for lp, tok_id in zip(topk.values, topk.indices):
                            tok_id  = tok_id.item()
                            new_tok = tokens + [tok_id]
                            if tok_id == END:
                                completed.append((score + lp.item(), new_tok[:-1]))
                            else:
                                new_beams.append((score + lp.item(), new_tok, new_h))
                    if not new_beams: break
                    new_beams.sort(key=lambda x: x[0] / max(len(x[1]), 1), reverse=True)
                    beams = new_beams[:beam_width]
                for score, tokens, _ in beams:
                    completed.append((score, tokens))

        completed.sort(key=lambda x: x[0] / max(len(x[1]), 1), reverse=True)
        seen, results = set(), []
        for _, tokens in completed:
            clean   = [t for t in tokens if t not in (END, PAD)]
            decoded = self.tokenizer.decode(clean)
            if decoded not in seen:
                seen.add(decoded)
                results.append(decoded)
        return results

    def get_logits(self, src_text):
        self.model.eval()
        src   = torch.tensor([self.tokenizer.encode(src_text)]).to(self.device)
        START = self.tokenizer.stoi['[START]']
        END   = self.tokenizer.stoi['[END]']
        PAD   = self.tokenizer.stoi['[PAD]']
        all_logits = []

        with torch.no_grad():
            if self.model_type == 'transformer':
                tgt = torch.tensor([[START]]).to(self.device)
                for _ in range(self.max_len):
                    logits = self.model(src, tgt)[0, -1, :]
                    all_logits.append(logits)
                    next_tok = logits.argmax(-1).item()
                    if next_tok == END:
                        break
                    tgt = torch.cat([tgt, torch.tensor([[next_tok]]).to(self.device)], dim=1)
            else:  # GRU
                src_mask = (src == PAD)
                encoder_outputs, hidden = self.model.encode(src)
                last_tok = START
                for _ in range(self.max_len):
                    dec_input = torch.tensor([last_tok]).to(self.device)
                    logits, hidden, _ = self.model.decode_step(
                        dec_input, hidden, encoder_outputs, src_mask
                    )
                    logits = logits[0]
                    all_logits.append(logits)
                    last_tok = logits.argmax(-1).item()
                    if last_tok == END:
                        break

        return torch.stack(all_logits)

    def evaluate(self, reactions, num_examples=10, print_samples=20, beam_width=3):
        from tqdm.notebook import tqdm
        preds, actuals, beam_cands = [], [], []
        print(f'\nEvaluating [{self.model_type}] on {num_examples} examples  (beam_width={beam_width})\n' + '-'*95)
        print(f"{'Reactants':<30} | {'Predicted Products':<30} | {'Actual Products':<30} | Match")
        print('-' * 95)
        # Single beam search per sample — reuse candidates for both predict and top-k
        for i, r in enumerate(tqdm(reactions[:num_examples], desc=f'Evaluating [{self.model_type}]', unit='rxn')):
            actual = r['output']
            cands  = self.predict_topk(r['input'], beam_width=beam_width)   # run beam search ONCE
            pred   = cands[0] if cands else ''        # best candidate = top-1
            preds.append(pred)
            actuals.append(actual)
            beam_cands.append(cands)
            if i < print_samples:
                match = '\u2713' if pred == actual else '\u2717'
                print(f"{r['input'][:30]:<30} | {pred[:30]:<30} | {actual[:30]:<30} | {match}")
        print('-' * 95)
        metrics = compute_metrics(preds, actuals, beam_cands)
        print_metrics(self.model_type, metrics)
        return metrics