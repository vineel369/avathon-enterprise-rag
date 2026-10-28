"""Download only the pinned BGE tokenizer; no model weights or paid APIs."""
import hashlib
import urllib.request

from scripts.chunking.build_chunks import ROOT, MODEL

REVISION = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
SHA256 = "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"


def main():
    path = ROOT / "data/tokenizers/bge-base-en-v1.5" / REVISION / "tokenizer.json"
    if path.exists():
        data = path.read_bytes()
    else:
        with urllib.request.urlopen(f"https://huggingface.co/{MODEL}/resolve/{REVISION}/tokenizer.json", timeout=120) as response:
            data = response.read()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError("Pinned tokenizer checksum mismatch")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    print(path)


if __name__ == "__main__":
    main()
