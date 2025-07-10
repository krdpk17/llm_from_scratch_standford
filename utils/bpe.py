from collections import Counter, defaultdict
from typing import List, Tuple

class BPE:
    def __init__(self, num_merges: int = 1000):
        self.num_merges = num_merges
        self.bpe_merges: List[Tuple[str, str]] = []

    def get_stats(self, vocab):
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs

    def merge_vocab(self, pair, vocab):
        new_vocab = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        for word in vocab:
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = vocab[word]
        return new_vocab

    def fit(self, corpus: List[str]):
        vocab = Counter([' '.join(word) + ' </w>' for word in corpus])
        for _ in range(self.num_merges):
            pairs = self.get_stats(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            vocab = self.merge_vocab(best, vocab)
            self.bpe_merges.append(best)

    def encode_word(self, word: str) -> List[str]:
        word = list(word) + ['</w>']
        while True:
            pairs = [(word[i], word[i+1]) for i in range(len(word)-1)]
            mergeable = [pair for pair in pairs if pair in self.bpe_merges]
            if not mergeable:
                break
            merge_pair = None
            for merge in self.bpe_merges:
                if merge in pairs:
                    merge_pair = merge
                    break
            if merge_pair is None:
                break
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word)-1 and (word[i], word[i+1]) == merge_pair:
                    new_word.append(word[i] + word[i+1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
        if word[-1] == '</w>':
            word = word[:-1]
        return word

    def encode(self, text: str) -> List[List[str]]:
        return [self.encode_word(word) for word in text.strip().split()]

    def decode_word(self, tokens: List[str]) -> str:
        """Reconstruct the original word from BPE tokens."""
        word = ''.join(tokens)
        if word.endswith('</w>'):
            word = word[:-4]
        return word

    def decode(self, encoded: List[List[str]]) -> List[str]:
        """Decode a list of encoded words back to their original form."""
        return [self.decode_word(tokens) for tokens in encoded]

if __name__ == "__main__":
    corpus = ["low", "lowest", "newer", "wider"]
    bpe = BPE(num_merges=10)
    bpe.fit(corpus)
    print("\nEncoding 'lowest newer':")
    original_text = "lowest newer"
    encoded = bpe.encode(original_text)
    print(encoded)
    print("\nDecoding:")
    decoded = bpe.decode(encoded)
    print(decoded)
    # Test: compare original text with decoded one
    decoded_text = ' '.join(decoded)
    print("\nTest passed:" if decoded_text == original_text else f"\nTest failed: '{decoded_text}' != '{original_text}'") 