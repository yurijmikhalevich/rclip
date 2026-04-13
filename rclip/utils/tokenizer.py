"""
Vendored and simplified CLIP BPE tokenizer.
Original source: open_clip (MIT License), which in turn is based on OpenAI's CLIP tokenizer.
Modified to remove the torch dependency: returns numpy int64 arrays instead of torch.LongTensor.
"""

import gzip
import html
import re
from functools import lru_cache
from typing import List, Tuple, Union

import ftfy
import numpy as np
import numpy.typing as npt
import regex

CONTEXT_LENGTH = 77


@lru_cache()
def _bytes_to_unicode():
  byte_values = (
    list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
  )
  code_points = byte_values[:]
  next_extra_code_point = 0
  for byte_value in range(2**8):
    if byte_value not in byte_values:
      byte_values.append(byte_value)
      code_points.append(2**8 + next_extra_code_point)
      next_extra_code_point += 1
  unicode_chars = [chr(code_point) for code_point in code_points]
  return dict(zip(byte_values, unicode_chars))


def _get_pairs(word: Tuple[str, ...]) -> set[Tuple[str, str]]:
  pairs: set[Tuple[str, str]] = set()
  prev_char = word[0]
  for char in word[1:]:
    pairs.add((prev_char, char))
    prev_char = char
  return pairs


def _basic_clean(text: str) -> str:
  text = ftfy.fix_text(text)
  text = html.unescape(html.unescape(text))
  return text.strip()


def _whitespace_clean(text: str) -> str:
  text = " ".join(text.split())
  return text.strip()


class SimpleTokenizer:
  def __init__(self, bpe_path: str):
    self.byte_encoder = _bytes_to_unicode()
    self.byte_decoder = {encoded_char: byte_value for byte_value, encoded_char in self.byte_encoder.items()}
    merges = gzip.open(bpe_path).read().decode("utf-8").split("\n")
    merges = merges[1 : 49152 - 256 - 2 + 1]
    merges = [tuple(merge.split()) for merge in merges]
    vocab = list(_bytes_to_unicode().values())
    vocab = vocab + [token_value + "</w>" for token_value in vocab]
    for merge in merges:
      vocab.append("".join(merge))
    special_tokens = ["<start_of_text>", "<end_of_text>"]
    vocab.extend(special_tokens)
    self.encoder = dict(zip(vocab, range(len(vocab))))
    self.decoder = {token_id: token for token, token_id in self.encoder.items()}
    self.bpe_ranks = dict(zip(merges, range(len(merges))))
    self.cache = {special_token: special_token for special_token in special_tokens}
    special_token_pattern = "|".join(re.escape(special_token) for special_token in special_tokens)
    self.pat = regex.compile(
      special_token_pattern + r"""|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
      regex.IGNORECASE,
    )
    self.sot_token_id = self.encoder["<start_of_text>"]
    self.eot_token_id = self.encoder["<end_of_text>"]
    self.context_length = CONTEXT_LENGTH

  def bpe(self, token: str) -> str:
    if token in self.cache:
      return self.cache[token]
    word = tuple(token[:-1]) + (token[-1] + "</w>",)
    pairs = _get_pairs(word)

    if not pairs:
      return token + "</w>"

    while True:
      bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))  # type: ignore[reportUnknownLambdaType]
      if bigram not in self.bpe_ranks:
        break
      first, second = bigram
      merged_word = []
      word_index = 0
      while word_index < len(word):
        try:
          match_index = word.index(first, word_index)
          merged_word.extend(word[word_index:match_index])
          word_index = match_index
        except Exception:
          merged_word.extend(word[word_index:])
          break

        if word[word_index] == first and word_index < len(word) - 1 and word[word_index + 1] == second:
          merged_word.append(first + second)
          word_index += 2
        else:
          merged_word.append(word[word_index])
          word_index += 1
      word = tuple(merged_word)
      if len(word) == 1:
        break
      else:
        pairs = _get_pairs(word)
    word = " ".join(word)
    self.cache[token] = word
    return word

  def encode(self, text: str) -> List[int]:
    bpe_tokens = []
    text = _whitespace_clean(_basic_clean(text)).lower()
    for token in regex.findall(self.pat, text):
      token = "".join(self.byte_encoder[byte_value] for byte_value in token.encode("utf-8"))
      bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" "))
    return bpe_tokens

  def __call__(self, texts: Union[str, List[str]], context_length: int = CONTEXT_LENGTH) -> npt.NDArray[np.int64]:
    """Tokenize input string(s) and return a numpy int64 array of shape [N, context_length]."""
    if isinstance(texts, str):
      texts = [texts]

    all_tokens = [[self.sot_token_id] + self.encode(text) + [self.eot_token_id] for text in texts]
    result = np.zeros((len(all_tokens), context_length), dtype=np.int64)

    for row_index, tokens in enumerate(all_tokens):
      if len(tokens) > context_length:
        tokens = tokens[:context_length]
        tokens[-1] = self.eot_token_id
      result[row_index, : len(tokens)] = tokens

    return result
