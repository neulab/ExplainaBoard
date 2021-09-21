import unittest
import os
import explainaboard.explainaboard_main as em


class IntegrationTest(unittest.TestCase):
    '''
    Does integration tests of end-to-end running of ExplainaBoard
    '''
    def __init__(self, *args, **kwargs):
        super(IntegrationTest, self).__init__(*args, **kwargs)
        path_file = os.path.dirname(__file__)
        self.example_dir = os.path.join(path_file, os.pardir, 'example')

    def test_absa_single(self):
        em.run_explainaboard('absa', [os.path.join(self.example_dir, 'test-laptop.tsv')], os.devnull, is_print_ece=True, is_print_case=True)

    def test_chunk_single(self):
        # TODO: currently testing is_print_case is too slow, so it's turned off but probably shouldn't be
        em.run_explainaboard('chunk', [os.path.join(self.example_dir, 'test-conll00.tsv')], os.devnull, is_print_case=False)

    def test_cws_single(self):
        # TODO: currently testing is_print_case is too slow, so it's turned off but probably shouldn't be
        em.run_explainaboard('cws', [os.path.join(self.example_dir, 'test-ctb.tsv')], os.devnull, is_print_case=False)

    def test_ner_single(self):
        # TODO: currently testing is_print_case is too slow, so it's turned off but probably shouldn't be
        em.run_explainaboard('ner', [os.path.join(self.example_dir, 'test-conll03.tsv')], os.devnull, is_print_case=False)

    def test_nli_single(self):
        em.run_explainaboard('nli', [os.path.join(self.example_dir, 'test-snli.tsv')], os.devnull, is_print_ece=True, is_print_case=True)

    def test_pos_single(self):
        # TODO: currently testing is_print_case is too slow, so it's turned off but probably shouldn't be
        em.run_explainaboard('pos', [os.path.join(self.example_dir, 'test-ptb2.tsv')], os.devnull, is_print_case=False)

    # TODO: There is no example for relation extraction?
    # def test_re_single(self):
    #   em.run_explainaboard('re', [os.path.join(self.example_dir, 'test-XXX.tsv')], os.devnull)

    def test_tc_single(self):
        em.run_explainaboard('tc', [os.path.join(self.example_dir, 'test-atis.tsv')], os.devnull, is_print_ece=True, is_print_case=True)

if __name__ == '__main__':
    unittest.main()
