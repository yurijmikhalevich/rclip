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

_CONTEXT_LENGTH = 77


@lru_cache()
def _bytes_to_unicode():
  bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
  cs = bs[:]
  n = 0
  for b in range(2**8):
    if b not in bs:
      bs.append(b)
      cs.append(2**8 + n)
      n += 1
  cs = [chr(n) for n in cs]
  return dict(zip(bs, cs))


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
    self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
    merges = gzip.open(bpe_path).read().decode("utf-8").split("\n")
    merges = merges[1 : 49152 - 256 - 2 + 1]
    merges = [tuple(merge.split()) for merge in merges]
    vocab = list(_bytes_to_unicode().values())
    vocab = vocab + [v + "</w>" for v in vocab]
    for merge in merges:
      vocab.append("".join(merge))
    special_tokens = ["<start_of_text>", "<end_of_text>"]
    vocab.extend(special_tokens)
    self.encoder = dict(zip(vocab, range(len(vocab))))
    self.decoder = {v: k for k, v in self.encoder.items()}
    self.bpe_ranks = dict(zip(merges, range(len(merges))))
    self.cache = {t: t for t in special_tokens}
    special = "|".join(re.escape(t) for t in special_tokens)
    self.pat = regex.compile(
      special + r"""|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
      regex.IGNORECASE,
    )
    self.sot_token_id = self.encoder["<start_of_text>"]
    self.eot_token_id = self.encoder["<end_of_text>"]
    self.context_length = _CONTEXT_LENGTH

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
      new_word = []
      i = 0
      while i < len(word):
        try:
          j = word.index(first, i)
          new_word.extend(word[i:j])
          i = j
        except Exception:
          new_word.extend(word[i:])
          break

        if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
          new_word.append(first + second)
          i += 2
        else:
          new_word.append(word[i])
          i += 1
      new_word = tuple(new_word)
      word = new_word
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
      token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
      bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" "))
    return bpe_tokens

  def __call__(self, texts: Union[str, List[str]], context_length: int = _CONTEXT_LENGTH) -> npt.NDArray[np.int64]:
    """Tokenize input string(s) and return a numpy int64 array of shape [N, context_length]."""
    if isinstance(texts, str):
      texts = [texts]

    all_tokens = [[self.sot_token_id] + self.encode(text) + [self.eot_token_id] for text in texts]
    result = np.zeros((len(all_tokens), context_length), dtype=np.int64)

    for i, tokens in enumerate(all_tokens):
      if len(tokens) > context_length:
        tokens = tokens[:context_length]
        tokens[-1] = self.eot_token_id
      result[i, : len(tokens)] = tokens

    return result
