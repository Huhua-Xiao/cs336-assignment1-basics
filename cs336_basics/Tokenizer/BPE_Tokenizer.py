
import re
import tokenize
import regex
from typing import List
from dataclasses import dataclass
from typing import Optional, overload, Iterable, Iterator
from collections import Counter, defaultdict
from functools import partial
import multiprocessing
import tqdm
import os
from typing import BinaryIO


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def get_chunk(input_paths: str, desired_num_chunks: int):
    chunk = [] 
    with open(input_paths, "rb") as f:
        boundries = find_chunk_boundaries(f, desired_num_chunks, b"<|endoftext|>")
        for start, end in zip(boundries[:-1], boundries[1:]):
            f.seek(start)
            chunk.append(f.read(end - start).decode("utf-8", errors="ignore"))

    return chunk

# === Tokenizer Training===

GPT2_TOKENIZER_REGEX = (
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)

# Define the parameters for BPE training
@dataclass
class BPETrainerParams:
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]

# Define the BPE trainer class
class BPETrainer:
    def __init__(self, params: Optional[BPETrainerParams] = None):
        if params:
            self.vocab = params.vocab
            self.merges = params.merges
        else:
            self.vocab = {}
            self.merges = []

    # def process_text_with_pre_tokenize(self, text: str):
    #     pre_tokenized_text = regex.findall(GPT2_TOKENIZER_REGEX, text)
    #     return pre_tokenized_text
    
    # def convert_to_byte_tuples(self,tokens):
    #     byte_tuples = []
    #     for token in tokens:
    #         token_bytes = token.encode("utf-8")
    #         byte_tuple = tuple(bytes([b]) for b in token_bytes)
    #         byte_tuples.append(byte_tuple)
    #     return byte_tuples
    
    # def count_frequencies(self, byte_tuples):
    #     count = Counter()
    #     for byte_tuple in byte_tuples:
    #         count[byte_tuple] += 1
    #     return count
    
    # def process_text_with_train_data(self, train_data: str):
    #     pre_tokenized_text = self.process_text_with_pre_tokenize(train_data)
    #     byte_tuples = self.convert_to_byte_tuples(pre_tokenized_text)
    #     frequencies = self.count_frequencies(byte_tuples)
    #     return frequencies
    
    # I found this method online, this method is more efficiency
    def process_text_with_train_data(self, text: str) -> Counter[tuple[bytes, ...]]:
        '''
        Pre-tokenizes text using GPT-2 regex, encodes tokens in UTF-8, and returns a Counter
        of token byte tuples (e.g., (b't', b'h', b'e')) with their frequencies.
        '''
        PAT = GPT2_TOKENIZER_REGEX
        tokens_counter = Counter()

        for match in regex.finditer(PAT, text):
            token = match.group()
            token_bytes = token.encode("utf-8")
            byte_tuple = tuple(bytes([b]) for b in token_bytes)  # tuple of bytes
            tokens_counter[byte_tuple] += 1

        return tokens_counter
    
    @staticmethod
    def process_special_tokens(origin_text: str, special_tokens: list[str]):
        split_pattern = re.compile("|".join(re.escape(token) for token in special_tokens))
        new_text = split_pattern.split(origin_text)
        tokenizer = BPETrainer()
        tokenize_counter = Counter()
        for text in new_text:
            text_counter = tokenizer.process_text_with_train_data(text)
            tokenize_counter.update(text_counter)
        return tokenize_counter

    def count_pairs_in_frequencies(self, frequencies: Counter[tuple[bytes, ...]]):
        dc = defaultdict(int)
        for token, freq in frequencies.items():
            for i in range(len(token) - 1):
                dc[token[i], token[i + 1]] += freq
        return dc
        
    def biggest_pair(self, dc: defaultdict):
        return max(dc, key=lambda x: (dc[x], x))
    
    def merge_biggest_pair(self, frequencies: Counter[tuple[bytes, ...]], pair: tuple[bytes, bytes]):
        # print(pair)
        new_frequencies = Counter()
        combined_pair = pair[0] + pair[1]
        for token, freq in frequencies.items():
            new_token = []
            index = 0
            while index < len(token):
                if index < len(token) - 1 and token[index] == pair[0] and token[index + 1] == pair[1]:
                    new_token.append(combined_pair)
                    index += 2
                else:
                    new_token.append(token[index])
                    index += 1
            new_frequencies[tuple(new_token)] += freq
        return new_frequencies

    def train(self, train_data: str, vocab_size: int, special_tokens: list[str]):
        self.vocab = {}
        self.merges = []

        offset = len(special_tokens)
        for i, token in enumerate(special_tokens):
            self.vocab[i] = token.encode("utf-8")
        for i in range(256):
            self.vocab[i + offset] = bytes([i])
        current_index = offset + 256

        chunks = get_chunk(train_data, multiprocessing.cpu_count())
        partical_func = partial(BPETrainer.process_special_tokens, special_tokens=special_tokens)
        with multiprocessing.Pool() as pool:
            results = list(pool.imap(partical_func, chunks))
        
        tokenize_counter = Counter()
        for result in results:
            tokenize_counter.update(result)
        
        with tqdm.tqdm(total=vocab_size - len(self.vocab), desc="Start to train BPE") as pbar:
            while len(self.vocab) < vocab_size:
                dc = self.count_pairs_in_frequencies(tokenize_counter)
                if not dc:
                    break

                pair = self.biggest_pair(dc)
                tokenize_counter = self.merge_biggest_pair(tokenize_counter, pair)
                self.vocab[current_index] = pair[0] + pair[1]
                self.merges.append(pair)
                current_index += 1
                pbar.update(1)
        
        return self.vocab, self.merges
    

class BPETokenizer: 
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.byte_vocab = vocab.copy()
        self.merges = merges
        self.special_tokens = special_tokens or []

        self.byte_to_id = {v: k for k, v in self.byte_vocab.items()}

        self.speical_tokens_bytes = [s.encode("utf-8") for s in self.special_tokens]
        self.special_tokens_set = set(self.speical_tokens_bytes)

        for token in self.speical_tokens_bytes:
            if token not in self.byte_to_id:
                new_id = len(self.byte_vocab)
                self.byte_vocab[new_id] = token
                self.byte_to_id[token] = new_id
        
        self.merges = [(a,b) for a, b in merges]
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        import json

        with open(vocab_filepath, "r") as vf:
            vocab_data = json.load(vf)
            vocab = {int(i): bytes(v, "latin1") for v, i in vocab_data.items()}

        merges = []
        with open(merges_filepath, "r") as mf:
            for line in mf:
                if line.strip() and not line.startswith("#"):
                    parts = line.strip().split()
                    if len(parts) == 2:
                        merges.append((bytes(parts[0], "latin1"), bytes(parts[1], "latin1")))

        return cls(vocab, merges, special_tokens)
    
    def _pre_tokenize(self, text: str):
        PAT = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        return regex.findall(PAT, text)

    def _byte_pair_merge(self, token: bytes):
        word = [bytes([b]) for b in token]
        pairs = lambda w: set((w[i], w[i+1]) for i in range(len(w) - 1))

        while True:
            candidate_pairs = pairs(word)
            ranked_pairs = [(self.merge_ranks[p], p) for p in candidate_pairs if p in self.merge_ranks]
            if not ranked_pairs:
                break
            _, best_pair = min(ranked_pairs)
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == best_pair[0] and word[i+1] == best_pair[1]:
                    new_word.append(word[i] + word[i+1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
        return word


    def encode(self, text: str):
        result = []
        special_pattern = "|".join(re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True))
        split_pattern = re.compile(f"({special_pattern})") if special_pattern else None

        segments = re.split(split_pattern, text) if split_pattern else [text]

        for segment in tqdm.tqdm(segments, desc="Encoding segments"):
            if segment == "":
                continue
            b = segment.encode("utf-8")
            if b in self.special_tokens_set:
                result.append(self.byte_to_id[b])
            else:
                for token in self._pre_tokenize(segment):
                    for merged in self._byte_pair_merge(token.encode("utf-8")):
                        result.append(self.byte_to_id[merged])
        
        return result

    def encode_iterable(self, iterable: Iterable[str]):
        for line in iterable:
            yield from self.encode(line)

    def decode(self, ids: list[int]):
        byte_seq = b"".join(self.byte_vocab[i] for i in ids)
        return byte_seq.decode("utf-8", errors="replace")