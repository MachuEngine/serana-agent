"""Deterministic offline embedding for tests: hashed bag of words."""

import hashlib
import re

from chromadb.api.types import EmbeddingFunction

DIM = 64


class FakeEmbedding(EmbeddingFunction):
    def __init__(self):
        pass

    def __call__(self, input):
        out = []
        for text in input:
            vec = [0.0] * DIM
            for word in re.findall(r"\w+", text.lower()):
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
            out.append(vec)
        return out

    @staticmethod
    def name() -> str:
        return "fake-embedding"

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(config):
        return FakeEmbedding()
