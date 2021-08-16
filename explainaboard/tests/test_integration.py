import unittest
import os
import explainaboard.explainaboard_main as em


class IntegrationTest(unittest.TestCase):
  '''
  Does integration tests of end-to-end running of ExplainaBoard
  '''
  def test_absa_single(self):
    em.run_explainaboard('absa', ['explainaboard/example/test-laptop.tsv'], os.devnull)

  def test_chunk_single(self):
    em.run_explainaboard('chunk', ['explainaboard/example/test-conll00.tsv'], os.devnull)

  def test_cws_single(self):
    em.run_explainaboard('cws', ['explainaboard/example/test-ctb.tsv'], os.devnull)

  def test_ner_single(self):
    em.run_explainaboard('ner', ['explainaboard/example/test-conll03.tsv'], os.devnull)

  def test_nli_single(self):
    em.run_explainaboard('nli', ['explainaboard/example/test-snli.tsv'], os.devnull)

  def test_pos_single(self):
    em.run_explainaboard('pos', ['explainaboard/example/test-ptb2.tsv'], os.devnull)

  # TODO: There is no example for relation extraction?
  # def test_re_single(self):
  #   em.run_explainaboard('re', ['explainaboard/example/test-XXX.tsv'], os.devnull)

  def test_tc_single(self):
    em.run_explainaboard('tc', ['explainaboard/example/test-atis.tsv'], os.devnull)

if __name__ == '__main__':
  unittest.main()
