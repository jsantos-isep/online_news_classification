import pytest
from online_news_classification.convert_datasets.the_guardian_to_csv import remove_html_tags, convert
import os
from dotenv import load_dotenv

class TestTheGuardianToCSV:

    @pytest.fixture(scope="session", autouse=True)
    def load_test_env(self):
        # Load environment variables from .env.test
        load_dotenv(dotenv_path=".env.test")
        # Optionally print or check if the environment variable is loaded correctly
        print(f"Loaded TEST_ENV={os.getenv('TEST_ENV')}")

    def test_remove_html_tags_with_html(self):
        assert remove_html_tags("<h1>Hello World</h1>"), "Hello World" # add assertion here


    def test_remove_html_tags_without_html(self):
       assert remove_html_tags("Hello World"), "Hello World"

    def test_convert(self):
        filepath = "tests/mocks/the_guardian_test_output.csv"
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"File {filepath} deleted.")
        convert(os.path.basename("the_guardian_test.json"), "the_guardian_test_output")
        assert os.path.exists(filepath), f"File {filepath} does not exist"