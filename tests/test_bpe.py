import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils')))
from bpe import BPE

def test_bpe_encode_decode():
    corpus = ["low", "lowest", "newer", "wider"]
    bpe = BPE(num_merges=10)
    bpe.fit(corpus)
    original_text = "lowest newer"
    encoded = bpe.encode(original_text)
    decoded = bpe.decode(encoded)
    decoded_text = ' '.join(decoded)
    assert decoded_text == original_text, f"Decoded text '{decoded_text}' does not match original '{original_text}'"

if __name__ == "__main__":
    test_bpe_encode_decode()
    print("BPE encode-decode test passed.") 