import traceback
import unittest

import yaml

from llm4dfm.pipeline.utils import output_as_valid_yaml
from tests.yaml_parsing.parsing_utils import load_parsable_datasets, load_unparsable_datasets, \
    store_parsing_test_output, load_unparsable_exercise, load_parsable_exercise


class ParsingTest(unittest.TestCase):

    def test_parsing(self):
        parsable_files = load_parsable_datasets()
        unparsable_files = load_unparsable_datasets()

        output_generated = dict()
        output_expected = dict()

        for file_name in unparsable_files:
            file = load_unparsable_exercise(file_name)

            for out in file['output']:
                self.assertRaises(yaml.YAMLError, lambda: output_as_valid_yaml(out))

        for file_name in parsable_files:
            file = load_parsable_exercise(file_name)

            output_generated[file_name] = []
            output_expected[file_name] = []

            for out, expected in zip(file['output'], file['expected']):

                try:
                    parsed_yaml = output_as_valid_yaml(out)
                    output_generated[file_name].append(parsed_yaml)
                    output_expected[file_name].append(expected)
                except Exception as e:
                    self.fail(f"Unexpected exception while parsing: {e}")

        for file in output_generated.keys():
            store_parsing_test_output(output_generated[file], file)

        for file in output_expected.keys():
            for idx, (expected, generated) in enumerate(zip(output_expected[file], output_generated[file])):
                try:
                    self.assertEqual(expected, generated)
                except:
                    print(f'\n[DEBUG] Error in file *{file}* {idx}-th output')
                    traceback.print_exc()
                    raise AssertionError


if __name__ == '__main__':
    unittest.main()
