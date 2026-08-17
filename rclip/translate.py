import difflib
import os
import sys
from pathlib import Path
from typing import Optional, Set


class LanguagePackageError(Exception):
  """Raised by ensure_language_installed() when the requested language package can't be
  obtained; str(error) is a user-facing message, including "did you mean" suggestions when the
  language code doesn't match any package available in the index."""


_translate_libs_available: Optional[bool] = None
_installed_source_languages: Set[str] = set()


def is_available() -> bool:
  global _translate_libs_available
  if _translate_libs_available is None:
    try:
      import argostranslate.package  # noqa: F401
      import argostranslate.translate  # noqa: F401

      _translate_libs_available = True
    except ImportError:
      _translate_libs_available = False
  return _translate_libs_available


def _is_ascii(text: str) -> bool:
  return all(ord(char) < 128 for char in text)


def _get_system_language() -> Optional[str]:
  """Best-effort OS/user locale language code (e.g. "ru" from "ru_RU.UTF-8")."""
  import locale

  raw = None
  for env_var in ("LANGUAGE", "LC_ALL", "LC_MESSAGES", "LANG"):
    raw = os.environ.get(env_var)
    if raw:
      break
  if not raw:
    try:
      raw = locale.getdefaultlocale()[0]
    except Exception:
      raw = None
  if not raw:
    return None
  # locale strings look like "ru_RU.UTF-8" or "ru:en"; take the first language subtag
  return raw.split(":")[0].split(".")[0].split("_")[0].lower() or None


def resolve_forced_lang(raw: Optional[str]) -> Optional[str]:
  """Resolves the raw "--lang" CLI value. Absent (None) stays None -- queries then only translate
  when a package for the system locale's language happens to already be installed. A bare
  "--lang" (raw == "") resolves to the system locale's language, so the caller can eagerly
  install it. An explicit code (raw == "ru") passes through unchanged."""
  if raw is None:
    return None
  if raw == "":
    return _get_system_language()
  return raw


def _has_installed_package(lang: str) -> bool:
  if lang in _installed_source_languages:
    return True
  if not is_available():
    return False

  import argostranslate.package

  installed = any(
    package.from_code == lang and package.to_code == "en" for package in argostranslate.package.get_installed_packages()
  )
  if installed:
    _installed_source_languages.add(lang)
  return installed


def _update_package_index() -> None:
  import socket

  import argostranslate.package

  # argostranslate.package.update_package_index() calls urllib.request.urlopen() with no
  # timeout, so a slow/unreachable index host hangs this call forever with no feedback.
  previous_timeout = socket.getdefaulttimeout()
  socket.setdefaulttimeout(15)
  try:
    argostranslate.package.update_package_index()
  finally:
    socket.setdefaulttimeout(previous_timeout)


def _download_argos_package(url: str, src_lang: str) -> Path:
  """Downloads an .argosmodel package with a visible progress bar (argostranslate's own
  Package.download()/install_package_for_language_pair() fetch the whole file silently)."""
  import tempfile

  import requests
  from tqdm import tqdm

  response = requests.get(url, stream=True, timeout=60)
  response.raise_for_status()
  total_bytes = int(response.headers.get("Content-Length", 0))

  with tempfile.NamedTemporaryFile(suffix=".argosmodel", delete=False) as tmp_file:
    tmp_path = Path(tmp_file.name)
    with tqdm(
      total=total_bytes or None,
      unit="B",
      unit_scale=True,
      desc=f'Downloading translation package for "{src_lang}"',
    ) as progress_bar:
      for chunk in response.iter_content(chunk_size=1024 * 256):
        tmp_file.write(chunk)
        progress_bar.update(len(chunk))

  return tmp_path


def ensure_language_installed(lang: str) -> None:
  """Downloads and installs the lang -> "en" argos-translate package, unless it's installed
  already. Raises LanguagePackageError -- with "did you mean" suggestions when lang doesn't match
  any package in the index -- if the package can't be obtained."""
  if _has_installed_package(lang):
    return

  import argostranslate.package

  print(f'rclip: fetching translation package index for "{lang}"...', file=sys.stderr)
  try:
    _update_package_index()
  except Exception as error:
    raise LanguagePackageError(f'rclip: could not reach the translation package index: {error}') from error

  en_targets = [package for package in argostranslate.package.get_available_packages() if package.to_code == "en"]
  available_package = next((package for package in en_targets if package.from_code == lang), None)

  if available_package is None or not available_package.links:
    known_codes = sorted({package.from_code for package in en_targets})
    suggestions = difflib.get_close_matches(lang, known_codes, n=3, cutoff=0.4)
    hint = f'; did you mean: {", ".join(suggestions)}?' if suggestions else ""
    raise LanguagePackageError(f'rclip: no translation package found for language "{lang}"{hint}')

  tmp_path = _download_argos_package(available_package.links[0], lang)
  try:
    argostranslate.package.install_from_path(tmp_path)
  finally:
    os.remove(tmp_path)
  _installed_source_languages.add(lang)


def _as_sentence(text: str) -> str:
  """argos-translate's small NMT models are trained on punctuated sentences and can mangle bare
  noun phrases -- dropping or merging words -- which is exactly the kind of short phrase rclip
  queries usually are. Capitalizing and terminating the phrase like a full sentence (e.g. turning
  the query "gato negro de noche" into "Gato negro de noche." before translating) makes these
  translate more reliably."""
  stripped = text.strip()
  if not stripped:
    return stripped
  sentence = stripped[0].upper() + stripped[1:]
  if sentence[-1] not in ".!?":
    sentence += "."
  return sentence


def translate_to_english(text: str, forced_lang: Optional[str] = None) -> str:
  """Translates a text query to English so it can be fed into rclip's English-only CLIP model.
  forced_lang (set via "--lang") is used as the source language unconditionally, even for
  ASCII-only text -- many languages (German, Italian, French, ...) are frequently written without
  any non-ASCII characters, and the user has already told us what language this is. Without
  forced_lang, the query is translated only when a package for the system locale's language is
  already installed (from a previous "--lang" run), so a plain search never triggers a network
  call or download; ASCII text is also assumed to already be English in that case, since it can't
  be told apart from a same-script foreign phrase without real language detection.
  Returns the original text unchanged on English input, missing optional dependencies, an
  uninstalled language package, or any translation failure."""
  if forced_lang is None and _is_ascii(text):
    return text
  if not is_available():
    return text

  src_lang = forced_lang or _get_system_language()
  if not src_lang or src_lang == "en":
    return text

  if not _has_installed_package(src_lang):
    return text

  import argostranslate.translate

  try:
    return argostranslate.translate.translate(_as_sentence(text), src_lang, "en")
  except Exception:
    return text
