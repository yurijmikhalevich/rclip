from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

from rclip import main as main_module
from rclip.utils import helpers, terminal


def test_interactive_terminal_requires_tty():
  supported, reason = terminal.is_interactive_terminal_supported(env={}, platform="linux", is_atty=False)
  assert not supported
  assert reason == "interactive mode requires a terminal (stdout is not a TTY)"


def test_preview_terminal_requires_tty():
  supported, reason = terminal.is_preview_terminal_supported(env={}, platform="linux", is_atty=False)
  assert not supported
  assert reason == "image preview requires a terminal (stdout is not a TTY)"


@pytest.mark.parametrize(
  "env",
  [
    {"KITTY_WINDOW_ID": "1"},
    {"KITTY_PID": "42"},
    {"TERM": "xterm-kitty"},
    {"GHOSTTY_RESOURCES_DIR": "/path/to/ghostty"},
    {"TERM_PROGRAM": "ghostty"},
    {"TERM": "xterm-ghostty"},
    {"TERM_PROGRAM": "iTerm.app", "TERM_PROGRAM_VERSION": "3.5.0"},
    {"TERM_PROGRAM": "iTerm.app", "TERM_PROGRAM_VERSION": "3.5.2"},
    {"TERM_PROGRAM": "iTerm.app"},
  ],
)
def test_interactive_terminal_supports_full_kitty_terms(env: dict[str, str]):
  supported, reason = terminal.is_interactive_terminal_supported(env=env, platform="darwin", is_atty=True)
  assert supported
  assert reason is None


@pytest.mark.parametrize(
  "env",
  [
    {"TERM_PROGRAM": "WezTerm"},
    {"WEZTERM_EXECUTABLE": "/path/to/wezterm"},
    {"WEZTERM_PANE": "1"},
    {"KONSOLE_VERSION": "22.04"},
  ],
)
def test_interactive_terminal_rejects_baseline_only_terms(env: dict[str, str]):
  supported, reason = terminal.is_interactive_terminal_supported(env=env, platform="linux", is_atty=True)
  assert not supported
  assert reason is not None
  assert "Unicode placeholders" in reason
  assert "--force-interactive" in reason


@pytest.mark.parametrize(
  "env",
  [
    {"TERM_PROGRAM": "WezTerm"},
    {"WEZTERM_EXECUTABLE": "/path/to/wezterm"},
    {"WEZTERM_PANE": "1"},
    {"KONSOLE_VERSION": "22.04"},
    {"KITTY_WINDOW_ID": "1"},
    {"GHOSTTY_RESOURCES_DIR": "/path"},
    {"TERM_PROGRAM": "iTerm.app", "TERM_PROGRAM_VERSION": "3.5.0"},
  ],
)
def test_preview_terminal_supports_baseline_kitty_terms(env: dict[str, str]):
  supported, reason = terminal.is_preview_terminal_supported(env=env, platform="linux", is_atty=True)
  assert supported
  assert reason is None


def test_interactive_terminal_rejects_old_iterm2():
  env = {"TERM_PROGRAM": "iTerm.app", "TERM_PROGRAM_VERSION": "3.4.19"}
  supported, reason = terminal.is_interactive_terminal_supported(env=env, platform="darwin", is_atty=True)
  assert not supported
  assert reason is not None
  assert "prior to 3.5.0" in reason
  assert "--force-interactive" in reason


def test_preview_terminal_rejects_old_iterm2():
  env = {"TERM_PROGRAM": "iTerm.app", "TERM_PROGRAM_VERSION": "3.4.19"}
  supported, reason = terminal.is_preview_terminal_supported(env=env, platform="darwin", is_atty=True)
  assert not supported
  assert reason is not None
  assert "prior to 3.5.0" in reason
  assert "--force-preview" in reason


def test_interactive_terminal_rejects_windows_terminal():
  env = {"WT_SESSION": "abcd-1234"}
  supported, reason = terminal.is_interactive_terminal_supported(env=env, platform="win32", is_atty=True)
  assert not supported
  assert reason is not None
  assert "Windows Terminal does not support" in reason
  assert "--force-interactive" in reason


def test_preview_terminal_rejects_windows_terminal():
  env = {"WT_SESSION": "abcd-1234"}
  supported, reason = terminal.is_preview_terminal_supported(env=env, platform="win32", is_atty=True)
  assert not supported
  assert reason is not None
  assert "Windows Terminal does not support" in reason
  assert "--force-preview" in reason


def test_interactive_terminal_rejects_windows_conhost():
  supported, reason = terminal.is_interactive_terminal_supported(env={}, platform="win32", is_atty=True)
  assert not supported
  assert reason is not None
  assert "Windows Command Prompt and PowerShell" in reason
  assert "--force-interactive" in reason


def test_preview_terminal_rejects_windows_conhost():
  supported, reason = terminal.is_preview_terminal_supported(env={}, platform="win32", is_atty=True)
  assert not supported
  assert reason is not None
  assert "Windows Command Prompt and PowerShell" in reason
  assert "--force-preview" in reason


def test_interactive_terminal_rejects_apple_terminal():
  env = {"TERM_PROGRAM": "Apple_Terminal"}
  supported, reason = terminal.is_interactive_terminal_supported(env=env, platform="darwin", is_atty=True)
  assert not supported
  assert reason is not None
  assert "Apple Terminal" in reason
  assert "--force-interactive" in reason


def test_preview_terminal_rejects_apple_terminal():
  env = {"TERM_PROGRAM": "Apple_Terminal"}
  supported, reason = terminal.is_preview_terminal_supported(env=env, platform="darwin", is_atty=True)
  assert not supported
  assert reason is not None
  assert "Apple Terminal" in reason
  assert "--force-preview" in reason


def test_interactive_terminal_rejects_alacritty():
  env = {"TERM": "alacritty"}
  supported, reason = terminal.is_interactive_terminal_supported(env=env, platform="linux", is_atty=True)
  assert not supported
  assert reason is not None
  assert "Alacritty" in reason
  assert "--force-interactive" in reason


def test_interactive_terminal_rejects_tmux_without_outer_vars():
  env = {"TMUX": "/tmp/tmux-1000/default,1234,0"}
  supported, reason = terminal.is_interactive_terminal_supported(env=env, platform="linux", is_atty=True)
  assert not supported
  assert reason is not None
  assert "running inside tmux" in reason
  assert "allow-passthrough" in reason
  assert "--force-interactive" in reason


def test_interactive_terminal_accepts_tmux_with_outer_kitty_vars():
  env = {"TMUX": "/tmp/tmux-1000/default,1234,0", "KITTY_WINDOW_ID": "1"}
  supported, reason = terminal.is_interactive_terminal_supported(env=env, platform="linux", is_atty=True)
  assert supported
  assert reason is None


def test_check_interactive_terminal_or_exit_exits_on_unsupported(capsys: pytest.CaptureFixture[str]):
  with pytest.raises(SystemExit) as exc_info:
    terminal.check_interactive_terminal_or_exit(env={}, platform="win32", is_atty=True)
  assert exc_info.value.code == 1
  captured = capsys.readouterr()
  assert "Windows Command Prompt and PowerShell" in captured.err


def test_check_preview_terminal_or_exit_exits_on_unsupported(capsys: pytest.CaptureFixture[str]):
  with pytest.raises(SystemExit) as exc_info:
    terminal.check_preview_terminal_or_exit(env={}, platform="win32", is_atty=True)
  assert exc_info.value.code == 1
  captured = capsys.readouterr()
  assert "Windows Command Prompt and PowerShell" in captured.err


def test_check_interactive_terminal_or_exit_succeeds_on_supported():
  env = {"KITTY_WINDOW_ID": "1"}
  terminal.check_interactive_terminal_or_exit(env=env, platform="darwin", is_atty=True)


def test_check_preview_terminal_or_exit_succeeds_on_supported():
  env = {"WEZTERM_EXECUTABLE": "/path/to/wezterm"}
  terminal.check_preview_terminal_or_exit(env=env, platform="darwin", is_atty=True)


# CLI Parsing and Flag Validation Tests


def test_force_interactive_flag_enables_interactive():
  parser = helpers.init_arg_parser()
  args = parser.parse_args(["--force-interactive"])
  assert args.force_interactive is True


def test_force_preview_flag_enables_preview():
  parser = helpers.init_arg_parser()
  args = parser.parse_args(["cat", "--force-preview"])
  assert args.force_preview is True


def test_main_rejects_unsupported_interactive_terminal_before_setup(monkeypatch: pytest.MonkeyPatch):
  monkeypatch.setattr(sys, "argv", ["rclip", "--interactive"])
  monkeypatch.setattr(terminal, "is_interactive_terminal_supported", lambda **_kwargs: (False, "unsupported term"))
  monkeypatch.setattr(main_module, "init_rclip", lambda **_options: pytest.fail("must reject before init_rclip"))

  with pytest.raises(SystemExit) as exc_info:
    main_module.main()
  assert exc_info.value.code == 1


def test_main_rejects_unsupported_preview_terminal_before_setup(monkeypatch: pytest.MonkeyPatch):
  monkeypatch.setattr(sys, "argv", ["rclip", "cat", "--preview"])
  monkeypatch.setattr(terminal, "is_preview_terminal_supported", lambda **_kwargs: (False, "unsupported term"))
  monkeypatch.setattr(main_module, "init_rclip", lambda **_options: pytest.fail("must reject before init_rclip"))

  with pytest.raises(SystemExit) as exc_info:
    main_module.main()
  assert exc_info.value.code == 1


def test_main_force_interactive_bypasses_terminal_check(monkeypatch: pytest.MonkeyPatch):
  init_options: dict[str, object] = {}
  tui_args: list[object] = []
  resources = tuple(SimpleNamespace(close=lambda: None) for _ in range(3))

  def fake_init_rclip(**options: object):
    init_options.update(options)
    return resources

  monkeypatch.setattr(sys, "argv", ["rclip", "--force-interactive"])
  monkeypatch.setattr(main_module, "is_snap", lambda: False)
  monkeypatch.setattr(
    terminal,
    "check_interactive_terminal_or_exit",
    lambda **_kwargs: pytest.fail("should be bypassed"),
  )
  monkeypatch.setattr(main_module, "init_rclip", fake_init_rclip)
  monkeypatch.setattr("rclip.tui.run_tui", lambda *args: tui_args.extend(args))

  main_module.main()
  assert init_options["allow_cross_thread_db"] is True
  assert tui_args == [resources[0], str(Path.cwd()), 100]


def test_main_force_preview_bypasses_terminal_check(monkeypatch: pytest.MonkeyPatch):
  init_options: dict[str, object] = {}
  resources = (
    SimpleNamespace(close=lambda: None, search=lambda *args, **kwargs: []),
    SimpleNamespace(close=lambda: None),
    SimpleNamespace(close=lambda: None),
  )

  def fake_init_rclip(**options: object):
    init_options.update(options)
    return resources

  monkeypatch.setattr(sys, "argv", ["rclip", "cat", "--force-preview"])
  monkeypatch.setattr(main_module, "is_snap", lambda: False)
  monkeypatch.setattr(
    terminal,
    "check_preview_terminal_or_exit",
    lambda **_kwargs: pytest.fail("should be bypassed"),
  )
  monkeypatch.setattr(main_module, "init_rclip", fake_init_rclip)
  monkeypatch.setattr(main_module, "print_results", lambda _results, _args: None)

  main_module.main()


@pytest.mark.parametrize(
  "args",
  [
    ["--force-interactive", "--preview"],
    ["--force-preview", "--interactive"],
    ["--force-interactive", "--force-preview"],
    ["--force-interactive", "--filepath-only"],
    ["cat", "--force-preview", "--filepath-only"],
  ],
)
def test_main_rejects_conflicting_force_flags(args: list[str], monkeypatch: pytest.MonkeyPatch):
  monkeypatch.setattr(sys, "argv", ["rclip", *args])
  with pytest.raises(SystemExit):
    main_module.main()
