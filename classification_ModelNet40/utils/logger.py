"""Minimal TSV logger (resume-safe, no trailing tabs)."""
import os
from typing import List, Optional


class Logger:
    def __init__(self, fpath: str, title: Optional[str] = None, resume: bool = False):
        self.fpath, self.title = fpath, title
        self.names: List[str] = []
        os.makedirs(os.path.dirname(os.path.abspath(fpath)), exist_ok=True)
        if resume and os.path.exists(fpath):
            with open(fpath, "r", encoding="utf-8") as f:
                header = f.readline().rstrip("\n")
            self.names = header.split("\t") if header else []
            self.file = open(fpath, "a", encoding="utf-8")
        else:  # resume requested but no file yet -> start a fresh log
            self.file = open(fpath, "w", encoding="utf-8")

    def set_names(self, names: List[str]) -> None:
        if self.names:            # resumed: header already written
            if list(names) != self.names:
                print(f"[logger] warning: resumed log has columns {self.names}, new run uses {names}")
            return
        self.names = list(names)
        self.file.write("\t".join(self.names) + "\n")
        self.file.flush()

    def append(self, values) -> None:
        if len(values) != len(self.names):
            raise ValueError(f"logger expected {len(self.names)} values, got {len(values)}")
        out = [f"{v:.6f}" if isinstance(v, float) else str(v) for v in values]
        self.file.write("\t".join(out) + "\n")
        self.file.flush()

    def close(self) -> None:
        if not self.file.closed:
            self.file.close()
