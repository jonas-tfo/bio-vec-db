import argparse
import sys
import json
import torch
from transformers import T5Tokenizer, T5EncoderModel, BertTokenizer, BertModel

T5_MODEL   = "Rostlab/prot_t5_xl_half_uniref50-enc"
BERT_MODEL = "Rostlab/prot_bert"


def parse_fasta(path: str) -> list[str]:
    sequences = []
    current = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current))
                    current = []
            else:
                current.append(line)
    if current:
        sequences.append("".join(current))
    return sequences


def embed_sequence_t5(sequence: str, tokenizer, encoder, device: str) -> list[float]:
    seq_spaced = " ".join(list(sequence))
    tokenized = tokenizer(
        seq_spaced,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    input_ids     = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)

    with torch.no_grad():
        output = encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = output.last_hidden_state  # (1, seq_len, hidden_dim)

    # strip EOS token, mean pool over residues
    seq_len = len(sequence)
    embeddings = embeddings.squeeze(0)[:seq_len]  # (seq_len, hidden_dim)
    pooled = embeddings.mean(dim=0)               # (hidden_dim,)
    return pooled.float().cpu().tolist()

def embed_sequence_t5_batch(sequences, tokenizer, encoder, device) -> list[list[float]]:
    embeddings = []
    for sequence in sequences:
        embedding = embed_sequence_t5(sequence, tokenizer, encoder, device)
        embeddings.append(embedding)
    return embeddings


def embed_sequence_bert(sequence: str, tokenizer, encoder, device: str) -> list[float]:
    seq_spaced = " ".join(list(sequence))
    tokenized = tokenizer(
        seq_spaced,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    input_ids      = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)

    with torch.no_grad():
        output = encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = output.last_hidden_state  # (1, seq_len+2, hidden_dim)

    # strip [CLS] (index 0) and [SEP] (last index), mean pool over residues
    embeddings = embeddings.squeeze(0)[1:-1]  # (seq_len, hidden_dim)
    pooled = embeddings.mean(dim=0)           # (hidden_dim,)
    return pooled.float().cpu().tolist()

def embed_sequence_bert_batch(sequences, tokenizer, encoder, device) -> list[list[float]]:
    embeddings = []
    for sequence in sequences:
        embedding = embed_sequence_bert(sequence, tokenizer, encoder, device)
        embeddings.append(embedding)
    return embeddings


def main():
    parser = argparse.ArgumentParser(description="Embed a protein sequence")
    parser.add_argument("--sequence", default=None, help="amino acid sequence to embed")
    parser.add_argument("--fasta", default=None, help="fasta containing amino acid sequences to embed")
    parser.add_argument("--model",    default=None,  help="huggingface model name (overrides arch default)")
    parser.add_argument("--arch",     default="bert", choices=["t5", "bert"], help="model architecture")
    parser.add_argument("--device",   default=None,  help="cpu, cuda, or mps")
    args = parser.parse_args()

    if args.device is None:
        device = (
            "mps"  if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = args.device

    if args.sequence is None and args.fasta is None:
        print("error: provide --sequence or --fasta", file=sys.stderr)
        sys.exit(1)

    if args.arch == "bert":
        model_name = args.model or BERT_MODEL
        tokenizer  = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
        encoder    = BertModel.from_pretrained(model_name)
        encoder.to(device)
        encoder.eval()
        if args.fasta:
            sequences = parse_fasta(args.fasta)
            result = embed_sequence_bert_batch(sequences, tokenizer, encoder, device)
        else:
            result = embed_sequence_bert(args.sequence, tokenizer, encoder, device)
    else:
        model_name = args.model or T5_MODEL
        tokenizer  = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
        encoder    = T5EncoderModel.from_pretrained(model_name, torch_dtype=torch.float16)
        encoder.to(device)
        encoder.eval()
        if args.fasta:
            sequences = parse_fasta(args.fasta)
            result = embed_sequence_t5_batch(sequences, tokenizer, encoder, device)
        else:
            result = embed_sequence_t5(args.sequence, tokenizer, encoder, device)

    # single sequence -> flat list (Vec<f32>), batch -> nested list (Vec<Vec<f32>>)
    json.dump(result, sys.stdout)


if __name__ == "__main__":
    main()
