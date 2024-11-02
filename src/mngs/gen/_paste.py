"""
This script does XYZ.
"""

# Imports

# Functions
def paste():
    import textwrap

    import pyperclip

    try:
        clipboard_content = pyperclip.paste()
        clipboard_content = textwrap.dedent(clipboard_content)
        exec(clipboard_content)
    except Exception as e:
        print(f"Could not execute clipboard content: {e}")


if __name__ == "__main__":
    # Start
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(sys, plt)

    import ipdb

    ipdb.set_trace()
    print("hello")  # copy this to your clipboar
    mngs.gen.paste()

    # Close
    mngs.gen.close(CONFIG)

# EOF

"""
/ssh:ywatanabe@444:/home/ywatanabe/proj/entrance/mngs/gen/_paste.py
"""
