"""Lightweight TSV training logger."""

import os


class Logger:
    def __init__(self, fpath, title=None, resume=False):
        self.file = None
        self.resume = resume
        self.title = title or ""
        self.names = []
        self.numbers = {}
        if fpath is not None:
            if resume and os.path.isfile(fpath):
                with open(fpath, "r") as f:
                    lines = [line.strip() for line in f if line.strip()]
                if lines:
                    self.names = lines[0].split("\t")
                    self.numbers = {name: [] for name in self.names}
                    for line in lines[1:]:
                        for name, val in zip(self.names, line.split("\t")):
                            self.numbers[name].append(float(val))
                self.file = open(fpath, "a")
            else:
                self.file = open(fpath, "w")

    def set_names(self, names):
        if self.resume:
            return
        self.names = list(names)
        self.numbers = {name: [] for name in self.names}
        self.file.write("\t".join(self.names) + "\n")
        self.file.flush()

    def append(self, numbers):
        assert len(self.names) == len(numbers), "Numbers do not match names"
        for name, num in zip(self.names, numbers):
            self.file.write(f"{num:.6f}\t")
            self.numbers[name].append(num)
        self.file.write("\n")
        self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()
