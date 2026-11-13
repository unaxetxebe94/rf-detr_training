"""Exclusive lock over the cache and the view, so two runs never write the same files at once."""

import json
import os
import socket
import time
from pathlib import Path

LOCK_NAME = ".cached.lock"


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes

        # os.kill(pid, 0) would terminate the process on Windows: query its exit code instead.
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class CacheLock:
    """``O_EXCL`` lock file with PID/host; a lock left by a dead process on this host is taken over."""

    def __init__(self, directory: Path):
        self.path = Path(directory) / LOCK_NAME
        self._held = False

    def _owner(self) -> dict | None:
        try:
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            return {}

    def acquire(self) -> "CacheLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        info = {"pid": os.getpid(), "host": socket.gethostname(), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                owner = self._owner()
                stale = owner is None or (
                    owner.get("host") == info["host"] and isinstance(owner.get("pid"), int)
                    and not _pid_alive(owner["pid"])
                )
                if not stale:
                    raise RuntimeError(
                        f"La caché está bloqueada por otro proceso ({owner}). Si no hay ningún entrenamiento ni "
                        f"preparación en marcha, borra {self.path}."
                    ) from None
                print(f"Lock huérfano de un proceso terminado ({owner}); se reutiliza.")
                self.path.unlink(missing_ok=True)
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(info, f)
            self._held = True
            return self
        raise RuntimeError(f"No se pudo adquirir el lock {self.path}.")

    def release(self) -> None:
        if self._held:
            self.path.unlink(missing_ok=True)
            self._held = False

    def __enter__(self) -> "CacheLock":
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()
