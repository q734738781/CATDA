import os.path
from copy import copy
from io import StringIO

import cv2
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from paddleocr.ppstructure.utility import draw_structure_result

from PDF_TO_MD.settings import zoom_factor, debug_result
from PIL import Image
import bs4

def extract_subtitles(title_blocks):
    title_names = []
    for block in title_blocks:
        block_res = block['res']
        paragraph = ''
        for data in block_res:
            paragraph += data['text']
            paragraph += ' '
        title_names.append(paragraph)
    return title_names

def draw_layout_result(layout_result, original_img, save_img_path):
    font_path = './simfang.ttf'  # PaddleOCR���ṩ�����
    image = copy(original_img)
    im_show = draw_structure_result(image, layout_result, font_path=font_path)
    im_show = Image.fromarray(im_show)
    if not os.path.exists(os.path.dirname(save_img_path)):
        os.makedirs(os.path.dirname(save_img_path))
    im_show.save(save_img_path)

def extract_text_with_ppstructure(pdf_path, layout_ocr_engine):
    '''
    Utility function to parse a PDF using PP-Structure OCR.
    Return the textblocks and subtitles
    '''
    doc = fitz.open(pdf_path)
    all_text_blocks = []
    all_title_names = []
    all_ocr_res = []

    table_count = 0
    img_count = 0
    eqn_count = 0

    for page_num, page in enumerate(doc):
        page_texts = []
        pm = page.get_pixmap(matrix=fitz.fitz.Matrix(zoom_factor, zoom_factor), alpha=False)
        img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        layout_result = layout_ocr_engine(img)
        all_ocr_res.append(layout_result)
        # Draw the layout result
        if debug_result:
            draw_layout_result(layout_result, img, f'./debug_output/page_{page_num}.png')
        # Sort the layout result by types
        text_blocks = []
        title_blocks = []
        for block in layout_result:
            text_blocks.append(block)
            if block['type'] == 'title':
                title_blocks.append(block)

        # Sort the order and save text results, up to 2 columns
        image_width = pm.width
        left_bound = image_width / 2 * 0.9
        left_text_blocks = []
        right_text_blocks = []
        for block in text_blocks:
            block_bbox = block['bbox']
            if block_bbox[0] < left_bound:
                left_text_blocks.append(block)
            else:
                right_text_blocks.append(block)

        # Sort the text blocks by the top coordinate
        left_text_blocks = sorted(left_text_blocks, key=lambda x: x['bbox'][1])
        right_text_blocks = sorted(right_text_blocks, key=lambda x: x['bbox'][1])
        page_text_blocks = left_text_blocks + right_text_blocks
        # Extract the text in blocks
        for block in page_text_blocks:
            paragraph = ''
            block_res = block['res']
            # Identify if block is 'table'
            if block['type'] == 'table':
                html_table = block_res['html']
                df = pd.read_html(html_table)[0]
                # Fill nan with string 'N/A'
                df = df.fillna('N/A')
                # Save the table as a csv string
                csv_buffer = StringIO()
                df.to_csv(csv_buffer, index=False)
                csv_string = csv_buffer.getvalue()
                # Replace the '\r\n' with '\n' for better formatting
                csv_string = csv_string.replace('\r\n', '\n')
                paragraph += f'"""csv{table_count}\n'
                paragraph += csv_string
                paragraph += f'"""\n'
                table_count += 1

            elif block['type'] == 'figure':
                paragraph += f'"""img{img_count}\n'
                paragraph += ' '.join([data['text'] for data in block_res])
                paragraph += '\n'
                paragraph += f'"""\n'
                img_count += 1

            elif block['type'] == 'equation':
                paragraph += f'"""eqn{eqn_count}\n'
                paragraph += ' '.join([data['text'] for data in block_res])
                paragraph += '\n'
                paragraph += f'"""\n'
                eqn_count += 1

            elif block['type'] == 'title':
                paragraph += 'Section:'
                paragraph += ' '.join([data['text'] for data in block_res])
                paragraph.replace('\n', '')

            else:
                paragraph += ' '.join([data['text'] for data in block_res])

            page_texts.append((block['type'],paragraph))

        page_title_names = extract_subtitles(title_blocks)
        # Clear the empty text block
        page_texts = [x for x in page_texts if len(x[1]) > 0]
        page_titles = [x.strip() for x in page_title_names if len(x) > 0]

        all_text_blocks.append(page_texts)
        all_title_names.extend(page_titles)


    doc.close()
    return all_text_blocks, all_title_names, all_ocr_res
