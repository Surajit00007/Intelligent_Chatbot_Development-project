import math
import re
import unicodedata
from pathlib import Path
import torch
import torch.nn as nn

PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, TURN_TOKEN = "<pad>", "<bos>", "<eos>", "<unk>", "<turn>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, TURN_TOKEN]
PAD_ID, BOS_ID, EOS_ID, UNK_ID, TURN_ID = range(len(SPECIAL_TOKENS))
TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?|[0-9]+|[.!?,;:]")

def normalize_text(text):
    text = unicodedata.normalize("NFD", str(text))
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", text.lower().strip())

def tokenize_text(text):
    return TOKEN_PATTERN.findall(normalize_text(text))

class WordTokenizer:
    def __init__(self, word2idx):
        self.word2idx = {str(k): int(v) for k, v in word2idx.items()}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        for idx, token in enumerate(SPECIAL_TOKENS):
            if self.word2idx.get(token) != idx:
                raise ValueError(f"Incompatible tokenizer: {token} must have ID {idx}")
    def encode_text(self, text):
        return [self.word2idx.get(token, UNK_ID) for token in tokenize_text(text)]
    def encode_context(self, context, max_len=40):
        if isinstance(context, str):
            context = [context]
        ids = []
        for i, turn in enumerate(context):
            if i:
                ids.append(TURN_ID)
            ids.extend(self.encode_text(turn))
        return ids[-max_len:] or [UNK_ID]
    def decode(self, token_ids):
        skip = {PAD_ID, BOS_ID, EOS_ID, TURN_ID, UNK_ID}
        words = [self.idx2word.get(int(idx), "") for idx in token_ids if int(idx) not in skip]
        text = " ".join(word for word in words if word)
        text = re.sub(r"\s+([.!?,;:])", r"\1", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:1].upper() + text[1:] if text else ""

class DialogueTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_encoder_layers=2, num_decoder_layers=2,
                 dim_feedforward=256, dropout=0.1, max_seq_len=40):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.vocab_size, self.d_model, self.max_seq_len = vocab_size, d_model, max_seq_len
        self.src_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.src_position = nn.Embedding(max_seq_len, d_model)
        self.tgt_position = nn.Embedding(max_seq_len, d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.output_projection = nn.Linear(d_model, vocab_size)
        self._reset_parameters()
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        with torch.no_grad():
            self.src_embedding.weight[PAD_ID].zero_()
            self.tgt_embedding.weight[PAD_ID].zero_()
    def _embed(self, ids, token_embedding, position_embedding):
        batch, length = ids.shape
        if length > self.max_seq_len:
            raise ValueError(f"Sequence length {length} exceeds {self.max_seq_len}")
        positions = torch.arange(length, device=ids.device).unsqueeze(0).expand(batch, -1)
        return self.embedding_dropout(token_embedding(ids) * math.sqrt(self.d_model) + position_embedding(positions))
    def forward(self, source_ids, target_input_ids):
        src_pad = source_ids.eq(PAD_ID)
        tgt_pad = target_input_ids.eq(PAD_ID)
        length = target_input_ids.size(1)
        causal = torch.triu(torch.ones(length, length, dtype=torch.bool, device=target_input_ids.device), diagonal=1)
        out = self.transformer(
            src=self._embed(source_ids, self.src_embedding, self.src_position),
            tgt=self._embed(target_input_ids, self.tgt_embedding, self.tgt_position),
            tgt_mask=causal,
            src_key_padding_mask=src_pad,
            tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=src_pad,
        )
        return self.output_projection(out)

def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def safe_load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)

def load_chatbot(checkpoint_path):
    device = choose_device()
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Run training in the notebook first.")
    checkpoint = safe_load_checkpoint(checkpoint_path, device)
    config = checkpoint["model_config"]
    tokenizer = WordTokenizer(checkpoint["tokenizer_state"]["word2idx"])
    model = DialogueTransformer(vocab_size=len(tokenizer.word2idx), **config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, tokenizer, device, checkpoint

@torch.inference_mode()
def generate_response(model, tokenizer, context, device, model_config, context_turns=3,
                     max_new_tokens=25, temperature=0.8, top_k=40, top_p=0.90,
                     do_sample=True, repetition_penalty=1.12):
    if isinstance(context, str):
        context = [context]
    context = list(context)[-max(1, int(context_turns)):]
    source_ids = tokenizer.encode_context(context, max_len=model_config["max_seq_len"])
    source = torch.tensor([source_ids], dtype=torch.long, device=device)
    generated = [BOS_ID]
    max_steps = max(1, min(int(max_new_tokens), model_config["max_seq_len"] - 1))
    model.eval()
    for _ in range(max_steps):
        target = torch.tensor([generated], dtype=torch.long, device=device)
        logits = model(source, target)[0, -1].float()
        for forbidden in (PAD_ID, BOS_ID, TURN_ID, UNK_ID):
            logits[forbidden] = -float("inf")
        punctuation = {".", "!", "?", ",", ";", ":"}
        emitted_words = [tokenizer.idx2word.get(token_id, "") for token_id in generated[1:]]
        has_lexical_word = any(word and word not in punctuation for word in emitted_words)
        lexical_ids_exist = any(idx >= len(SPECIAL_TOKENS) and word not in punctuation for word, idx in tokenizer.word2idx.items())
        if not has_lexical_word and lexical_ids_exist:
            logits[EOS_ID] = -float("inf")
            for punctuation_token in punctuation:
                punctuation_id = tokenizer.word2idx.get(punctuation_token)
                if punctuation_id is not None:
                    logits[punctuation_id] = -float("inf")
        if repetition_penalty and repetition_penalty > 1:
            for token_id in set(generated[1:]):
                if token_id < logits.numel():
                    logits[token_id] = logits[token_id] * repetition_penalty if logits[token_id] < 0 else logits[token_id] / repetition_penalty
        if not do_sample or temperature <= 0:
            next_id = int(torch.argmax(logits).item())
        else:
            logits = logits / max(float(temperature), 1e-4)
            if top_k and 0 < top_k < logits.numel():
                cutoff = torch.topk(logits, int(top_k)).values[-1]
                logits[logits < cutoff] = -float("inf")
            if top_p and 0 < top_p < 1:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(-1)
                remove = cumulative > top_p
                remove[1:] = remove[:-1].clone()
                remove[0] = False
                logits[sorted_indices[remove]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            if not torch.isfinite(probs).all() or probs.sum() <= 0:
                next_id = int(torch.argmax(logits).item())
            else:
                next_id = int(torch.multinomial(probs, 1).item())
        if next_id == EOS_ID:
            break
        generated.append(next_id)
        if len(generated) >= model_config["max_seq_len"]:
            break
    return tokenizer.decode(generated[1:])
