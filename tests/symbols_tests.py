import unittest

from TTS.tts.utils.text import _phonemes

class SymbolsTest(unittest.TestCase):
    def test_uniqueness(self):  #pylint: disable=no-self-use
        assert sorted(_phonemes) == sorted(list(set(_phonemes))), " {} vs {} ".format(len(_phonemes), len(set(_phonemes)))
        