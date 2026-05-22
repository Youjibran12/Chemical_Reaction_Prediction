import torch
from evaluation.metrics import compute_metrics, print_metrics
from tqdm import tqdm

class EnsemblePredictor:


    def __init__(self, gru_predictor, tf_predictor,
                 gru_weight=0.4, transformer_weight=0.6):
        assert abs(gru_weight + transformer_weight - 1.0) < 1e-6, \
            'Weights must sum to 1.0'
        assert gru_predictor.tokenizer.stoi == tf_predictor.tokenizer.stoi, \
            'Tokenizer mismatch! Both models must share the same tokenizer.'

        self.gru       = gru_predictor
        self.tf        = tf_predictor
        self.w_g       = gru_weight
        self.w_t       = transformer_weight
        self.tokenizer = gru_predictor.tokenizer
        self.device    = gru_predictor.device
        self.max_len   = gru_predictor.max_len
        print(f'Ensemble ready  (GRU weight={gru_weight}, TF weight={transformer_weight})')


    def _ensemble_log_probs(self, src, gru_enc_out, gru_hidden,
                             tf_tgt_so_far, last_token, src_mask):
        dec_input = torch.tensor([last_token]).to(self.device)
        g_logits, new_hidden, _ = self.gru.model.decode_step(
            dec_input, gru_hidden, gru_enc_out, src_mask
        )
        g_log_probs = torch.log_softmax(g_logits[0], dim=-1)

        t_logits    = self.tf.model(src, tf_tgt_so_far)[0, -1, :]
        t_log_probs = torch.log_softmax(t_logits, dim=-1)

        combined = self.w_g * g_log_probs + self.w_t * t_log_probs
        return combined, new_hidden

    # Beam search decoding
    def predict(self, src_text, beam_width=5):
        self.gru.model.eval()
        self.tf.model.eval()

        src      = torch.tensor([self.tokenizer.encode(src_text)]).to(self.device)
        START    = self.tokenizer.stoi['[START]']
        END      = self.tokenizer.stoi['[END]']
        PAD      = self.tokenizer.stoi['[PAD]']
        src_mask = (src == PAD)

        with torch.no_grad():
            gru_enc_out, gru_hidden_init = self.gru.model.encode(src)
            tf_start  = torch.tensor([[START]]).to(self.device)
            beams     = [(0.0, [], gru_hidden_init, tf_start)]
            completed = []

            for _ in range(self.max_len):
                new_beams = []
                for score, tokens, g_hid, tf_tgt in beams:
                    last_tok = tokens[-1] if tokens else START
                    combined_lp, new_g_hid = self._ensemble_log_probs(
                        src, gru_enc_out, g_hid, tf_tgt, last_tok, src_mask
                    )
                    topk = combined_lp.topk(beam_width)
                    for lp, tok_id in zip(topk.values, topk.indices):
                        tok_id  = tok_id.item()
                        new_tok = tokens + [tok_id]
                        new_tf  = torch.cat(
                            [tf_tgt, torch.tensor([[tok_id]]).to(self.device)], dim=1
                        )
                        if tok_id == END:
                            completed.append((score + lp.item(), new_tok[:-1]))
                        else:
                            new_beams.append((score + lp.item(), new_tok, new_g_hid, new_tf))

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


    # Beam search — all hypotheses
    def predict_topk(self, src_text, beam_width=5):
        self.gru.model.eval()
        self.tf.model.eval()

        src      = torch.tensor([self.tokenizer.encode(src_text)]).to(self.device)
        START    = self.tokenizer.stoi['[START]']
        END      = self.tokenizer.stoi['[END]']
        PAD      = self.tokenizer.stoi['[PAD]']
        src_mask = (src == PAD)

        with torch.no_grad():
            gru_enc_out, gru_hidden_init = self.gru.model.encode(src)
            tf_start  = torch.tensor([[START]]).to(self.device)
            beams     = [(0.0, [], gru_hidden_init, tf_start)]
            completed = []

            for _ in range(self.max_len):
                new_beams = []
                for score, tokens, g_hid, tf_tgt in beams:
                    last_tok = tokens[-1] if tokens else START
                    combined_lp, new_g_hid = self._ensemble_log_probs(
                        src, gru_enc_out, g_hid, tf_tgt, last_tok, src_mask
                    )
                    topk = combined_lp.topk(beam_width)
                    for lp, tok_id in zip(topk.values, topk.indices):
                        tok_id  = tok_id.item()
                        new_tok = tokens + [tok_id]
                        new_tf  = torch.cat(
                            [tf_tgt, torch.tensor([[tok_id]]).to(self.device)], dim=1
                        )
                        if tok_id == END:
                            completed.append((score + lp.item(), new_tok[:-1]))
                        else:
                            new_beams.append((score + lp.item(), new_tok, new_g_hid, new_tf))

                if not new_beams: break
                new_beams.sort(key=lambda x: x[0] / max(len(x[1]), 1), reverse=True)
                beams = new_beams[:beam_width]

            for score, tokens, _, _ in beams:
                completed.append((score, tokens))

            completed.sort(key=lambda x: x[0] / max(len(x[1]), 1), reverse=True)

            results = []
            for _, tokens in completed:
                clean = [t for t in tokens if t not in (END, PAD)]
                results.append(self.tokenizer.decode(clean))

            seen, unique = set(), []
            for s in results:
                if s not in seen:
                    seen.add(s)
                    unique.append(s)
            return unique


    def evaluate(self, reactions, num_examples=50, print_samples=20, beam_width=3):
        from tqdm.notebook import tqdm
        preds, actuals, beam_cands = [], [], []
        print(f'\nEvaluating [Ensemble] on {num_examples} examples  (beam_width={beam_width})\n' + '-'*95)
        print(f"{'Reactants':<30} | {'Predicted Products':<30} | {'Actual Products':<30} | Match")
        print('-' * 95)
        # Single beam search per sample — reuse candidates for both predict and top-k
        for i, r in enumerate(tqdm(reactions[:num_examples], desc='Evaluating [Ensemble]', unit='rxn')):
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
        print_metrics('Ensemble (GRU + Transformer)', metrics)
        return metrics, preds, actuals