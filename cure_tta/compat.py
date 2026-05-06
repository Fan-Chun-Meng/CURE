import re as std_re
import sys
import types


def _install_ftfy_stub() -> None:
    try:
        import ftfy  # type: ignore  # noqa: F401
    except ImportError:
        ftfy_stub = types.ModuleType("ftfy")
        ftfy_stub.fix_text = lambda text: text
        sys.modules["ftfy"] = ftfy_stub


def _install_regex_stub() -> None:
    try:
        import regex  # type: ignore  # noqa: F401
    except ImportError:
        regex_stub = types.ModuleType("regex")

        def _rewrite_pattern(pattern: str) -> str:
            return pattern.replace(r"\p{L}", "A-Za-z").replace(r"\p{N}", "0-9")

        def _is_compiled_pattern(pattern) -> bool:
            return hasattr(pattern, "findall") and hasattr(pattern, "pattern")

        def _compile_regex(pattern, flags=0):
            if _is_compiled_pattern(pattern):
                return pattern
            return std_re.compile(_rewrite_pattern(pattern), flags)

        def _sub_regex(pattern, repl, string, count=0, flags=0):
            if _is_compiled_pattern(pattern):
                return pattern.sub(repl, string, count=count)
            return std_re.sub(_rewrite_pattern(pattern), repl, string, count=count, flags=flags)

        def _findall_regex(pattern, string, flags=0):
            if _is_compiled_pattern(pattern):
                return pattern.findall(string)
            return std_re.findall(_rewrite_pattern(pattern), string, flags)

        regex_stub.IGNORECASE = std_re.IGNORECASE
        regex_stub.compile = _compile_regex
        regex_stub.sub = _sub_regex
        regex_stub.findall = _findall_regex
        sys.modules["regex"] = regex_stub


def _install_gdown_stub() -> None:
    try:
        import gdown  # type: ignore  # noqa: F401
    except ImportError:
        gdown_stub = types.ModuleType("gdown")

        def _missing_gdown(*args, **kwargs):
            raise ImportError("gdown is not installed, but a download was requested.")

        gdown_stub.download = _missing_gdown
        sys.modules["gdown"] = gdown_stub


def _install_open_clip_stub() -> None:
    try:
        import open_clip  # type: ignore  # noqa: F401
    except ImportError:
        open_clip_stub = types.ModuleType("open_clip")
        open_clip_stub.get_tokenizer = lambda *args, **kwargs: None
        sys.modules["open_clip"] = open_clip_stub


def ensure_runtime_compat() -> None:
    _install_ftfy_stub()
    _install_regex_stub()
    _install_gdown_stub()
    _install_open_clip_stub()


ensure_runtime_compat()
