"""
Testing how inspect works

Needs to be run in a function if run in main it breaks ...
"""

import os
import inspect

input_file_g: str
output_file_g: str


def PDF2(input_file=None, output_file=None):
    frame_g = inspect.currentframe().f_back
    if input_file is None:
        input_file = frame_g.f_code.co_filename
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + '.pdf'

    print(f"Created PDF file: {output_file}")