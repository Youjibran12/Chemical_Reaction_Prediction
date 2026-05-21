SMILES_REGEX = re.compile(
    r'(\[[^\]]+\]'
    r'|Br|Cl|Si|Se|se'
    r'|@@|@'
    r'|%\d{2}'
    r'|.)'
)

def smiles_tokenize(text: str):
    return SMILES_REGEX.findall(text)


class SMILESTokenizer:
    def __init__(self):
        self.special_tokens = ['[PAD]', '[START]', '[END]', '[UNK]']
        self.vocab = self.special_tokens.copy()
        self.stoi  = {}
        self.itos  = {}

    def build_vocab(self, reactions):

        if self.stoi:
            print(f'Vocab already built ({len(self.vocab)} tokens) — skipping rebuild')
            return
        tokens = set()
        for r in reactions:
            tokens.update(smiles_tokenize(r['input'] + r['output']))
        self.vocab += sorted(tokens)
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}
        print(f'Vocab size: {len(self.vocab)}')

    def encode(self, text, max_len=128):
        toks   = smiles_tokenize(text)
        tokens = ([self.stoi['[START]']]
                  + [self.stoi.get(t, self.stoi['[UNK]']) for t in toks]
                  + [self.stoi['[END]']])
        tokens  = tokens[:max_len]
        tokens += [self.stoi['[PAD]']] * (max_len - len(tokens))
        return tokens

    def decode(self, tokens):
        skip = {self.stoi['[PAD]'], self.stoi['[START]'], self.stoi['[END]']}
        return ''.join(self.itos[t] for t in tokens if t not in skip)