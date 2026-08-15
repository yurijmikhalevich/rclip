import os
import sys
import types
from pathlib import Path
from typing import Callable

import pytest

import rclip.translate as translate_module


@pytest.fixture(autouse=True)
def reset_translate_state(monkeypatch: pytest.MonkeyPatch):
  monkeypatch.setattr(translate_module, "_translate_libs_available", None)
  monkeypatch.setattr(translate_module, "_installed_source_languages", set())


def _install_fake_argostranslate_package(monkeypatch: pytest.MonkeyPatch, **attrs: object) -> None:
  """Injects a fake argostranslate.package submodule; `import argostranslate.package` also imports
  the parent package first, so the parent needs the submodule attached, not just sys.modules."""
  fake_package_module = types.ModuleType("argostranslate.package")
  for name, value in attrs.items():
    setattr(fake_package_module, name, value)
  fake_argostranslate = types.ModuleType("argostranslate")
  setattr(fake_argostranslate, "package", fake_package_module)
  monkeypatch.setitem(sys.modules, "argostranslate", fake_argostranslate)
  monkeypatch.setitem(sys.modules, "argostranslate.package", fake_package_module)


def _install_fake_argostranslate_translate(
  monkeypatch: pytest.MonkeyPatch, translate_fn: Callable[[str, str, str], str]
) -> None:
  fake_translate_module = types.ModuleType("argostranslate.translate")
  setattr(fake_translate_module, "translate", translate_fn)
  fake_argostranslate = types.ModuleType("argostranslate")
  setattr(fake_argostranslate, "translate", fake_translate_module)
  monkeypatch.setitem(sys.modules, "argostranslate", fake_argostranslate)
  monkeypatch.setitem(sys.modules, "argostranslate.translate", fake_translate_module)


def test_as_sentence_capitalizes_and_adds_period():
  assert translate_module._as_sentence("gato negro de noche") == "Gato negro de noche."


def test_as_sentence_does_not_double_punctuate():
  assert translate_module._as_sentence("gato negro de noche?") == "Gato negro de noche?"
  assert translate_module._as_sentence("gato negro de noche!") == "Gato negro de noche!"
  assert translate_module._as_sentence("gato negro de noche.") == "Gato negro de noche."


def test_as_sentence_empty_string_stays_empty():
  assert translate_module._as_sentence("   ") == ""


def test_is_ascii_true_for_english():
  assert translate_module._is_ascii("cat on the couch") is True


def test_is_ascii_false_for_non_ascii_text():
  assert translate_module._is_ascii("un gato en el sofá") is False


def test_get_system_language_reads_lang_env_var(monkeypatch: pytest.MonkeyPatch):
  monkeypatch.delenv("LANGUAGE", raising=False)
  monkeypatch.delenv("LC_ALL", raising=False)
  monkeypatch.delenv("LC_MESSAGES", raising=False)
  monkeypatch.setenv("LANG", "es_ES.UTF-8")

  assert translate_module._get_system_language() == "es"


def test_get_system_language_prefers_language_env_var(monkeypatch: pytest.MonkeyPatch):
  monkeypatch.setenv("LANGUAGE", "fr:en")
  monkeypatch.setenv("LANG", "es_ES.UTF-8")

  assert translate_module._get_system_language() == "fr"


def test_get_system_language_falls_back_to_getdefaultlocale(monkeypatch: pytest.MonkeyPatch):
  monkeypatch.delenv("LANGUAGE", raising=False)
  monkeypatch.delenv("LC_ALL", raising=False)
  monkeypatch.delenv("LC_MESSAGES", raising=False)
  monkeypatch.delenv("LANG", raising=False)

  import locale

  monkeypatch.setattr(locale, "getdefaultlocale", lambda: ("ja_JP", "UTF-8"))

  assert translate_module._get_system_language() == "ja"


def test_get_system_language_none_when_undetermined(monkeypatch: pytest.MonkeyPatch):
  monkeypatch.delenv("LANGUAGE", raising=False)
  monkeypatch.delenv("LC_ALL", raising=False)
  monkeypatch.delenv("LC_MESSAGES", raising=False)
  monkeypatch.delenv("LANG", raising=False)

  import locale

  monkeypatch.setattr(locale, "getdefaultlocale", lambda: (None, None))

  assert translate_module._get_system_language() is None


def test_resolve_forced_lang_none_when_flag_absent():
  assert translate_module.resolve_forced_lang(None) is None


def test_resolve_forced_lang_uses_system_language_when_bare(monkeypatch: pytest.MonkeyPatch):
  monkeypatch.setattr(translate_module, "_get_system_language", lambda: "es")

  assert translate_module.resolve_forced_lang("") == "es"


def test_resolve_forced_lang_passes_through_explicit_code():
  assert translate_module.resolve_forced_lang("es") == "es"


def test_is_available_false_when_libs_not_installed(monkeypatch: pytest.MonkeyPatch):
  # setting a module to None in sys.modules forces the next "import name" to raise
  # ImportError, simulating the optional [translate] extra not being installed, even
  # though it's actually installed in this dev environment (for type-checking/tests).
  monkeypatch.setitem(sys.modules, "argostranslate", None)
  monkeypatch.setitem(sys.modules, "argostranslate.package", None)
  monkeypatch.setitem(sys.modules, "argostranslate.translate", None)

  assert translate_module.is_available() is False


def test_translate_to_english_noop_for_ascii_text():
  result = translate_module.translate_to_english("cat on the couch")

  assert result == "cat on the couch"


def test_translate_to_english_noop_when_libs_missing(monkeypatch: pytest.MonkeyPatch):
  monkeypatch.setattr(translate_module, "is_available", lambda: False)

  result = translate_module.translate_to_english("un gato en el sofá")

  assert result == "un gato en el sofá"


def test_translate_to_english_uses_system_language_without_forced_lang(monkeypatch: pytest.MonkeyPatch):
  monkeypatch.setattr(translate_module, "is_available", lambda: True)
  monkeypatch.setattr(translate_module, "_get_system_language", lambda: "es")
  monkeypatch.setattr(translate_module, "_has_installed_package", lambda _lang: True)

  calls: list[tuple[str, str, str]] = []

  def fake_translate(text: str, from_code: str, to_code: str) -> str:
    calls.append((text, from_code, to_code))
    return "cat on the couch"

  _install_fake_argostranslate_translate(monkeypatch, fake_translate)

  result = translate_module.translate_to_english("un gato en el sofá")

  assert result == "cat on the couch"
  # the query is turned into a full sentence before translation (see _as_sentence)
  assert calls == [("Un gato en el sofá.", "es", "en")]


def test_translate_to_english_skips_translation_when_system_language_package_not_installed(
  monkeypatch: pytest.MonkeyPatch,
):
  monkeypatch.setattr(translate_module, "is_available", lambda: True)
  monkeypatch.setattr(translate_module, "_get_system_language", lambda: "es")
  monkeypatch.setattr(translate_module, "_has_installed_package", lambda _lang: False)

  network_calls: list[None] = []
  _install_fake_argostranslate_package(
    monkeypatch,
    update_package_index=lambda: network_calls.append(None),
  )

  result = translate_module.translate_to_english("un gato en el sofá")

  # without --lang, a query in a not-yet-installed language never triggers a download attempt
  assert result == "un gato en el sofá"
  assert network_calls == []


def test_translate_to_english_forced_lang_overrides_system_language(monkeypatch: pytest.MonkeyPatch):
  monkeypatch.setattr(translate_module, "is_available", lambda: True)
  monkeypatch.setattr(translate_module, "_get_system_language", lambda: "en")
  monkeypatch.setattr(translate_module, "_has_installed_package", lambda lang: lang == "de")

  calls: list[tuple[str, str, str]] = []

  def fake_translate(text: str, from_code: str, to_code: str) -> str:
    calls.append((text, from_code, to_code))
    return "translated"

  _install_fake_argostranslate_translate(monkeypatch, fake_translate)

  result = translate_module.translate_to_english("Käse", forced_lang="de")

  assert result == "translated"
  assert calls == [("Käse.", "de", "en")]


def test_translate_to_english_forced_lang_translates_ascii_text(monkeypatch: pytest.MonkeyPatch):
  # regression test: German/Italian/French, etc. are frequently written with no non-ASCII
  # characters at all (e.g. "Ein schwarzer Hund im Park"), so the ASCII shortcut must not
  # short-circuit translation once the user has explicitly forced a source language via --lang.
  monkeypatch.setattr(translate_module, "is_available", lambda: True)
  monkeypatch.setattr(translate_module, "_has_installed_package", lambda lang: lang == "de")

  calls: list[tuple[str, str, str]] = []

  def fake_translate(text: str, from_code: str, to_code: str) -> str:
    calls.append((text, from_code, to_code))
    return "a black dog in the park"

  _install_fake_argostranslate_translate(monkeypatch, fake_translate)

  result = translate_module.translate_to_english("Ein schwarzer Hund im Park", forced_lang="de")

  assert result == "a black dog in the park"
  assert calls == [("Ein schwarzer Hund im Park.", "de", "en")]


def test_translate_to_english_returns_unchanged_when_no_system_language(monkeypatch: pytest.MonkeyPatch):
  monkeypatch.setattr(translate_module, "is_available", lambda: True)
  monkeypatch.setattr(translate_module, "_get_system_language", lambda: None)

  result = translate_module.translate_to_english("un gato en el sofá")

  assert result == "un gato en el sofá"


def test_translate_to_english_falls_back_on_translation_error(monkeypatch: pytest.MonkeyPatch):
  monkeypatch.setattr(translate_module, "is_available", lambda: True)
  monkeypatch.setattr(translate_module, "_has_installed_package", lambda _lang: True)

  def raising_translate(_text: str, _from_code: str, _to_code: str) -> str:
    raise RuntimeError("boom")

  _install_fake_argostranslate_translate(monkeypatch, raising_translate)

  result = translate_module.translate_to_english("un gato en el sofá", forced_lang="es")

  assert result == "un gato en el sofá"


def test_has_installed_package_caches_positive_result(monkeypatch: pytest.MonkeyPatch):
  monkeypatch.setattr(translate_module, "is_available", lambda: True)
  fake_package = types.SimpleNamespace(from_code="es", to_code="en")
  lookups: list[None] = []
  _install_fake_argostranslate_package(
    monkeypatch,
    get_installed_packages=lambda: lookups.append(None) or [fake_package],
  )

  assert translate_module._has_installed_package("es") is True
  assert translate_module._has_installed_package("es") is True
  assert len(lookups) == 1
  assert "es" in translate_module._installed_source_languages


def test_has_installed_package_false_when_not_installed(monkeypatch: pytest.MonkeyPatch):
  monkeypatch.setattr(translate_module, "is_available", lambda: True)
  _install_fake_argostranslate_package(monkeypatch, get_installed_packages=lambda: [])

  assert translate_module._has_installed_package("es") is False


def test_ensure_language_installed_skips_download_when_already_installed(monkeypatch: pytest.MonkeyPatch):
  fake_package = types.SimpleNamespace(from_code="es", to_code="en")
  update_calls: list[None] = []
  monkeypatch.setattr(translate_module, "is_available", lambda: True)
  _install_fake_argostranslate_package(
    monkeypatch,
    get_installed_packages=lambda: [fake_package],
    update_package_index=lambda: update_calls.append(None),
  )

  translate_module.ensure_language_installed("es")

  assert update_calls == []
  assert "es" in translate_module._installed_source_languages


def test_ensure_language_installed_downloads_missing_package(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
  fake_package = types.SimpleNamespace(from_code="es", to_code="en", links=["https://example.com/es_en.argosmodel"])
  install_calls: list[str] = []
  download_calls: list[tuple[str, str]] = []

  fake_downloaded_path = str(tmp_path / "fake.argosmodel")
  with open(fake_downloaded_path, "wb") as f:
    f.write(b"fake")

  monkeypatch.setattr(translate_module, "is_available", lambda: True)
  _install_fake_argostranslate_package(
    monkeypatch,
    get_installed_packages=lambda: [],
    update_package_index=lambda: None,
    get_available_packages=lambda: [fake_package],
    install_from_path=lambda path: install_calls.append(path),
  )
  monkeypatch.setattr(
    translate_module,
    "_download_argos_package",
    lambda url, src_lang: download_calls.append((url, src_lang)) or fake_downloaded_path,
  )

  translate_module.ensure_language_installed("es")

  assert download_calls == [("https://example.com/es_en.argosmodel", "es")]
  assert install_calls == [fake_downloaded_path]
  assert "es" in translate_module._installed_source_languages
  assert not os.path.exists(fake_downloaded_path)  # temp download file is cleaned up after install


def test_ensure_language_installed_raises_with_suggestions_for_unknown_code(monkeypatch: pytest.MonkeyPatch):
  es_package = types.SimpleNamespace(from_code="es", to_code="en", links=["https://example.com/es_en.argosmodel"])
  de_package = types.SimpleNamespace(from_code="de", to_code="en", links=["https://example.com/de_en.argosmodel"])
  monkeypatch.setattr(translate_module, "is_available", lambda: True)
  _install_fake_argostranslate_package(
    monkeypatch,
    get_installed_packages=lambda: [],
    update_package_index=lambda: None,
    get_available_packages=lambda: [es_package, de_package],
  )

  with pytest.raises(translate_module.LanguagePackageError) as exc_info:
    translate_module.ensure_language_installed("fs")

  message = str(exc_info.value)
  assert "fs" in message
  assert "es" in message


def test_ensure_language_installed_raises_when_index_unreachable(monkeypatch: pytest.MonkeyPatch):
  monkeypatch.setattr(translate_module, "is_available", lambda: True)

  def raising_update_package_index() -> None:
    raise OSError("network unreachable")

  _install_fake_argostranslate_package(
    monkeypatch,
    get_installed_packages=lambda: [],
    update_package_index=raising_update_package_index,
  )

  with pytest.raises(translate_module.LanguagePackageError):
    translate_module.ensure_language_installed("es")
