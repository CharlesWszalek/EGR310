# Python program to create a pdf file

from fpdf import FPDF
import os
import matplotlib.pyplot as plt
import numpy as np
import inspect

with open('output.txt', 'w') as file:
    pass  # Clear the file

def PDF(input_file = None, output_file = None, location = 0):
    if input_file is None:
        frame = inspect.currentframe().f_back
        input_file = frame.f_code.co_filename
        input_file = os.path.basename(input_file)

    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + '.pdf'

    image_address = os.path.splitext(output_file)[0]
    if location:
        print(f"File: {input_file}")
        print(f"Output: {output_file}")
        print(f"Address: {image_address}")

    # init pdf
    pdf = FPDF()

    # add header
    pdf.add_page()
    pdf.set_font("Times", size=12, style='B')
    pdf.cell(200, 5, txt='Charles Wszalek                                                                       \
                                                            EGR 310',ln=1, align='L')
    pdf.set_font("Times", size=24, style='B')
    pdf.cell(200, 20, txt='PYTHON CODE ', ln=1, align='C')

    # read code
    pdf.set_font("Arial", size=12)
    with open(input_file, 'r') as python_file:
        content = python_file.read()
    content = content.split('\n')

    # add code
    for x in content:
        pdf.cell(200, 5, txt=x, ln=1, align='L')

    # add header
    pdf.add_page()
    pdf.set_font("Times", size=24, style='B')
    pdf.cell(200, 10, txt='OUTPUT ', ln=1, align='C')
    pdf.set_font("Times", size=11, style='B')
    pdf.cell(200, 5, txt='Plots ', ln=1, align='C')

    # add images
    # if not os.path.exists(os.getcwd() + 'figures_'+image_address+'/'):
    #     os.makedirs(os.getcwd() + 'figures_'+image_address+'/')
    # folder_dir = os.getcwd() + 'figures_'+image_address+'/'
    if not os.path.exists('figures_'+image_address+'/'):
        os.makedirs('figures_'+image_address+'/')
    folder_dir = 'figures_' + image_address + '/'
    xoffset = 0
    yoffset = 0
    count = 0
    for items in sorted(os.listdir(folder_dir)):
        pdf.image(folder_dir + items, x=10 +xoffset, y=25 + yoffset, h=70)
        count += 1
        if count % 2 == 0:
            yoffset += 70
            xoffset = 0
        if count % 2 != 0:
            xoffset += 100
        if (count % 4 == 0) & (count != len(os.listdir(folder_dir))):
            xoffset = 0
            yoffset = 0
            pdf.add_page()

    y = pdf.get_y()
    y = y + ((count+1)/2)*60
    pdf.set_y(y)
    pdf.set_font("Times", size=11, style='B')
    pdf.cell(200, 20, txt='Prints ', ln=1, align='C')
    # add output
    pdf.set_font("Arial", size=11)
    if os.path.exists('output.txt'):
        with open('output.txt', 'r') as printed:
            content = printed.read()
        content = content.split('\n')
        for x in content:
            pdf.cell(200, 5, txt=x, ln=1, align='L')

    # save the pdf with name .pdf
    pdf.output(output_file)

def SAVE(number, input_file:str = None, output_file:str = None, location = 0):
    if input_file is None:
        frame = inspect.currentframe().f_back
        input_file = frame.f_code.co_filename
        input_file = os.path.basename(input_file)

    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + '.pdf'

    image_address = os.path.splitext(output_file)[0]
    if location:
        print(f"File: {input_file}")
        print(f"Output: {output_file}")
        print(f"Address: {image_address}, type: {type(image_address)}")

    if not os.path.exists('figures_'+image_address+'/'):
        os.makedirs('figures_'+image_address+'/')
    plt.savefig('figures_'+image_address+'/image'+str(number)+".jpg")

def print2(*args):
    with open("output.txt", "a") as f:
        print(*args, file=f)
        print(*args)
