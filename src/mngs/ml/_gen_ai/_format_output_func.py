#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-06-10 23:11:58 (ywatanabe)"
# /home/ywatanabe/proj/mngs/src/mngs/ml/_gen_ai/_format_output_func.py


"""
This script does XYZ.
"""


"""
Imports
"""
import re
import sys

import markdown2
import matplotlib.pyplot as plt
import mngs

# sys.path = ["."] + sys.path
# from scripts import utils, load

"""
Warnings
"""
# warnings.simplefilter("ignore", UserWarning)


"""
Config
"""
# CONFIG = mngs.gen.load_configs()


"""
Functions & Classes
"""


def format_output_func(out_text):
    def find_unwrapped_urls(text):
        # Regex to find URLs that are not already within <a href> tags
        url_pattern = (
            r'(?<!<a href=")(https?://|doi:|http://doi.org/)[^\s,<>"]+'
        )

        # Find all matches that are not already wrapped
        unwrapped_urls = re.findall(url_pattern, text)

        return unwrapped_urls

    def add_a_href_tag(text):
        # Function to replace each URL with its wrapped version
        def replace_url(match):
            url = match.group(0)
            # Normalize DOI URLs
            if url.startswith("doi:"):
                url = "https://doi.org/" + url[4:]
            return f'<a href="{url}">{url}</a>'

        # Regex pattern to match URLs not already wrapped in <a> tags
        url_pattern = (
            r'(?<!<a href=")(https?://|doi:|http://doi.org/)[^\s,<>"]+'
        )

        # Replace all occurrences of unwrapped URLs in the text
        updated_text = re.sub(url_pattern, replace_url, text)

        return updated_text

    def add_masked_api_key(text, api_key):
        masked_api_key = f"{api_key[:4]}****{api_key[-4:]}"
        return text + f"\n(API Key: {masked_api_key}"

    out_text = markdown2.markdown(out_text)
    out_text = add_a_href_tag(out_text)
    out_text = re.sub(r"^<p>(.*)</p>$", r"\1", out_text, flags=re.DOTALL)
    return out_text


def main():
    pass


if __name__ == "__main__":
    # # Argument Parser
    # import argparse
    # parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--var', '-v', type=int, default=1, help='')
    # parser.add_argument('--flag', '-f', action='store_true', default=False, help='')
    # args = parser.parse_args()

    # Main
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
