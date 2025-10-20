import base64
import json
import os
import re
import subprocess
import time
import traceback
import warnings
from io import StringIO
import cv2
import fitz
import pandas as pd
import requests
from PIL import Image
from PaperPreprocess.src.parseutils import is_image_based_pdf
from PaperPreprocess.src.direct_parser import extract_text_blocks_with_fitz
from PaperPreprocess.src.settings import (ocr_settings, predefined_section_names, substitute_special_char,
                                          azure_api_key,azure_api_endpoint, azure_auto_load, azure_output_dpi)

# Global variable to store the layout ocr engine
layout_ocr_engine = None
global_extractor = None

class BaseExtractor:
    def __init__(self):
        self.full_text = ''
        self.text_blocks = None
        self.ocr_res = None
        self.section_titles = predefined_section_names
        self.paper_section_titles = []

    def initialize(self, *args, **kwargs):
        # Initialize shared resources or check prerequisites
        raise NotImplementedError

    def parse(self, paperpath, save_dir):
        # Parse a specific document
        raise NotImplementedError

    def check_pdf_validaty(self):
        try:
            doc = fitz.open(self.paperpath)
            doc.close()
        except:
            raise Exception('Invalid PDF file')

    def get_page_nums(self):
        doc = fitz.open(self.paperpath)
        page_num = doc.page_count
        doc.close()
        return page_num


    def _check_create_dir(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    def _split_chapters(self):
        def is_any_substring_in_string(string_list, main_string):
            for _string in string_list:
                if main_string in _string:
                    return True
            return False

        # Find the chapter names, and split the full text into chapters
        # Use Three lines of \n as the chapter separator
        # Priority: Longer title first
        self.paper_section_titles = sorted(self.paper_section_titles,key=lambda x:len(x),reverse=True) # Sort the titles by length
        processed_titles_lower = []
        for title in self.paper_section_titles:
            if is_any_substring_in_string(processed_titles_lower,title.lower()):
                continue
            pattern = re.compile(re.escape(title))
            self.full_text,sub_count = pattern.subn('Section:' + title + '\n', self.full_text)
            if sub_count > 0:
                processed_titles_lower.append(title.lower())

    def save_txt(self, base_save_dir):
        txt_dir = os.path.join(base_save_dir, 'txt')
        self._check_create_dir(txt_dir)
        path = os.path.join(txt_dir, 'output.txt')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.full_text.strip())

class FitzExtractor(BaseExtractor):
    def initialize(self):
        # Fitz does not require specific initialization
        pass

    def _get_full_text(self):
        text = ''
        for page_blocks in self.text_blocks:
            page_blocks = sorted(page_blocks, key=lambda x: x[5])
            for block in page_blocks:
                text += block[4].replace("\n", " ")
                text += '\n'
            text += '\n\n\n' # Use three lines of \n as the page separator
        for key, value in substitute_special_char.items():
            text = text.replace(key, value)
        self.full_text = text

    def check_if_in_single_text_block(self,title):
        # Check if one blocks contents equals to title name:
        for page_block in self.text_blocks:
            for text_block in page_block:
                text = text_block[4]
                if text == title:
                    return True
        return False

    def figure_extract(self, base_save_dir):
        # Extract figures from the paper
        img_dir = os.path.join(base_save_dir, 'images')
        self._check_create_dir(img_dir)
        pdf_handle = fitz.open(self.paperpath)
        for page in pdf_handle:
            image_list = page.get_images()
            img_count = 0
            for image in image_list:
                # Use xref to get the image
                xref = image[0]
                pix = fitz.Pixmap(pdf_handle, xref)
                # Save the image
                img_save_path = os.path.join(img_dir, f'page{page.number}_img{img_count}.jpeg')
                pix.save(img_save_path)
                pix = None
                img_count += 1

    def table_extract(self, base_save_dir):
        table_dir = os.path.join(base_save_dir, 'tables')
        self._check_create_dir(table_dir)
        # Use camelot to extract tables
        import camelot
        tables = camelot.read_pdf(self.paperpath, pages='all')
        for table_num, table in enumerate(tables):
            save_path = os.path.join(table_dir, f'table{table_num}.csv')
            table.to_csv(save_path)


    def parse(self, paperpath, save_dir):
        # Reset the full text and text blocks
        self.full_text = ''
        self.text_blocks = None
        self.paper_section_titles = self.section_titles # No implementation to extract section titles using Fitz

        # Check if the paper metadata
        self.paperpath = paperpath
        self.save_dir = save_dir
        self.check_pdf_validaty()
        self.is_image_pdf = is_image_based_pdf(self.paperpath)
        if self.is_image_pdf:
            raise Exception('Fitz does not support image-based PDFs')
        self.page_num = self.get_page_nums()

        # Text extraction
        self.text_blocks = extract_text_blocks_with_fitz(self.paperpath)
        self._get_full_text()
        #self._split_chapters()
        self.save_txt(save_dir)
        # Figure extraction
        self.figure_extract(save_dir)
        # Table extraction
        self.table_extract(save_dir)

class PaddleExtractor(BaseExtractor):
    def initialize(self):
        from paddleocr import PPStructure
        global layout_ocr_engine
        if layout_ocr_engine is None:
            print('Loading the layout ocr engine')
            layout_ocr_engine = PPStructure(**ocr_settings)
            print('Layout ocr engine loaded')

    def _text_block_extract(self):
        from PaperPreprocess.src.image_parser import extract_text_with_ppstructure
        self.text_blocks, paper_section_titles, self.ocr_res = extract_text_with_ppstructure(self.paperpath,
                                                                                             layout_ocr_engine)
        # Extend the section titles
        paper_section_titles = [x for x in paper_section_titles if len(x) > 1]
        new_section_titles = []
        for section_title in paper_section_titles:
            if section_title[0].isupper() and len(section_title.split(' ')):
                # Only detect the section titles with more than 1 word
                new_section_titles.append(section_title)
        self.paper_section_titles = self.section_titles + new_section_titles

    def _get_full_text(self):
        text = ''
        for page_blocks in self.text_blocks:
            page_text = ''
            for block_index, block in enumerate(page_blocks):
                block_type, block_text = block
                is_special_block = block_type in ['table', 'figure', 'equation', 'title']
                before_special_block = False

                if is_special_block:
                    if block_type == 'table':
                        page_text += block_text
                    else:
                        page_text += block_text + '\n'
                else:
                    # Check if the next block is a special block
                    if block_index + 1 < len(page_blocks):
                        next_block_type, _ = page_blocks[block_index + 1]
                        if next_block_type in ['table', 'figure', 'equation']:
                            before_special_block = True

                    if before_special_block:
                        page_text += block_text + '\n'
                    else:
                        page_text += block_text + '\n\n'

            text += page_text + '\n'
            # Use three lines of \n as the page separator.
            # Since two \n already added in previous block, only add on here
        self.full_text = text

    def figure_extract(self, base_save_dir):
        img_dir = os.path.join(base_save_dir, 'images')
        self._check_create_dir(img_dir)
        img_count = 0
        for page_num, page_res in enumerate(self.ocr_res):
            caption_text = ''
            for block in page_res:
                if block['type'] == 'figure':
                    save_path = os.path.join(img_dir, f'page{page_num}_img{img_count}.png')
                    img_array = block['img']
                    cv2.imwrite(save_path, img_array)
                    img_count += 1
                elif block['type'] == 'figure_caption':
                    caption_text += ' '.join([x['text'] for x in block['res']]) + '\n'

            if caption_text:
                save_path = os.path.join(img_dir, f'page{page_num}_caption.txt')
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(caption_text.strip())

    def table_extract(self, base_save_dir):
        table_dir = os.path.join(base_save_dir, 'tables')
        self._check_create_dir(table_dir)
        table_count = 0
        for page_num, page_res in enumerate(self.ocr_res):
            caption_text = ''
            for block in page_res:
                if block['type'] == 'table':
                    html_table = block['res']['html']
                    df = pd.read_html(html_table)[0]
                    save_path = os.path.join(table_dir, f'page{page_num}_table{table_count}.csv')
                    df.to_csv(save_path, index = False)
                    # Save the img of table, use different engine to recognize that later
                    img_save_path = os.path.join(table_dir, f'page{page_num}_table{table_count}.png')
                    cv2.imwrite(img_save_path, block['img'])
                    table_count += 1

                elif block['type'] == 'table_caption':
                    caption_text += ' '.join([x['text'] for x in block['res']]) + '\n'

            if caption_text:
                save_path = os.path.join(table_dir, f'page{page_num}_caption.txt')
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(caption_text.strip())

    def eqn_extract(self, base_save_dir):
        eqn_dir = os.path.join(base_save_dir, 'equations')
        self._check_create_dir(eqn_dir)
        equation_count = 0
        for page_num,page_res in enumerate(self.ocr_res):
            for block in page_res:
                if block['type'] == 'equation':
                    save_path = os.path.join(eqn_dir,f'page{page_num}_equation{equation_count}.png')
                    img_array = block['img']
                    cv2.imwrite(save_path,img_array)
                    equation_count += 1


    def parse(self, paperpath, save_dir):
        # Reset the full text and text blocks
        self.full_text = ''
        self.text_blocks = None

        # Check if the paper metadata
        self.paperpath = paperpath
        self.save_dir = save_dir
        self.check_pdf_validaty()
        self.page_num = self.get_page_nums()

        # Text extraction
        self._text_block_extract()
        self._get_full_text()
        #self._split_chapters()
        self.save_txt(save_dir)
        # Figure extraction
        self.figure_extract(save_dir)
        # Table extraction
        self.table_extract(save_dir)
        # Eqn extraction
        self.eqn_extract(save_dir)

class NougatExtractor(BaseExtractor):
    def initialize(self, parse_mode='cli', API_address=None):
        self.parse_mode = parse_mode
        if parse_mode == 'cli':
            self._check_nougat_installation()
        elif parse_mode == 'API':
            self.API_address = API_address
            if self.API_address is None:
                raise Exception('Please provide the API address')

    def _check_nougat_installation(self):
        try:
            subprocess.check_output(['nougat','-h'])
        except:
            raise Exception('Nougat is not installed, please install it first')

    def nougat_parse(self, base_save_dir=None):
        # Use nougat to directly parse the paper into markdown
        # It uses cli for single paper parsing
        # For batch parsing, use API to avoid the overhead of loading the model
        if base_save_dir is None:
            base_save_dir = os.path.join(self.save_dir,'nougat')
        self._check_create_dir(base_save_dir)

        # Call the program
        # usage: nougat [-h] [--batchsize BATCHSIZE] [--checkpoint CHECKPOINT] [--model MODEL] [--out OUT] [--recompute] [--full-precision] [--no-markdown] [--markdown] [--no-skipping] [--pages PAGES] pdf [pdf ...]

        if self.parse_mode == 'cli':
            # Since each call would include the loading of the model,
            # it is not suitable for batch processing, and should be only used for test purpose
            try:
                subprocess.check_output(['nougat',self.paperpath,'--out',base_save_dir,'-m','0.1.0-base'])
            except:
                raise Exception('Nougat failed to parse the paper')
        elif self.parse_mode == 'API':
            # API Example usage by curl
            """
            curl -X 'POST' \
              'http://127.0.0.1:8503/predict/' \
              -H 'accept: application/json' \
              -H 'Content-Type: multipart/form-data' \
              -F 'file=@<PDFFILE.pdf>;type=application/pdf'
            """
            # Use requests to call the API
            requrest_url = self.API_address + '/predict/'
            headers = {'accept': 'application/json'}
            with open(self.paperpath,'rb') as f:
                paper_bytes = f.read()
            files = {'file': ('filename.pdf', paper_bytes, 'application/pdf')}
            response = requests.post(requrest_url, headers=headers, files=files)
            if response.status_code == 200:
                # Response is a base64 encoded markdown file string
                decoded_response = base64.b64decode(response.text)
                with open(os.path.join(base_save_dir,'output.md'),'wb') as f:
                    f.write(decoded_response)
            else:
                raise Exception('Nougat API failed to parse the paper')

    def parse(self, paperpath, save_dir):
        self.paperpath = paperpath
        self.save_dir = save_dir
        self.check_pdf_validaty()

        self.nougat_parse()

class AzureExtractor(BaseExtractor):
    def __init__(self):
        super().__init__()
        self.client = None
        self.result = None
        self.auto_load_json = None
        self.pdf_handle = None

    def initialize(self, endpoint, key, auto_load=True):
        from azure.core.credentials import AzureKeyCredential
        from azure.ai.documentintelligence import DocumentIntelligenceClient

        self.client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))
        self.auto_load_json = auto_load

    def poll_azure_result(self):
        from azure.ai.documentintelligence.models import AnalyzeResult
        with open(self.paperpath, "rb") as f:
            poller = self.client.begin_analyze_document(
                "prebuilt-layout", analyze_request=f, content_type="application/octet-stream"
            )
        self.result: AnalyzeResult = poller.result()

    def get_caption(self, figure_object):
        if figure_object.caption:
            return figure_object.caption.content if figure_object.caption.content else ''
        else:
            return ''

    def get_formatted_table(self, table_object):
        table_text = ''
        if table_object.caption:
            table_text += table_object.caption.content + '\n'
        if table_object.cells:
            # Create an empty dataframe with table_object.column_count columns and row_count rows
            table = pd.DataFrame(index=range(table_object.row_count), columns=range(table_object.column_count))
            for cell in table_object.cells:
                cell_col = cell.column_index
                cell_row = cell.row_index
                cell_content = cell.content
                table.iloc[cell_row, cell_col] = cell_content
            # Fill the empty cells with empty string
            # table.fillna('-', inplace=True)
            # Make the empty string also with '-'
            # table = table.map(lambda x: x if x != '' else '-')
            # Convert to csv string using StringIO
            csv_buffer = StringIO()
            table.to_csv(csv_buffer, index=False, sep = '|' )
            csv_text = csv_buffer.getvalue().replace('\r\n','\n')
            # Remove the first line, cause it is a fake header created by position
            csv_text = csv_text.split('\n')[1:]
            csv_text = '\n'.join(csv_text)
            table_text += csv_text
        if table_object.footnotes:
            for footnote in table_object.footnotes:
                if footnote.elements:
                    table_text+= footnote.content + '\n'

        return table_text

    def text_parsing(self):
        text_blocks = self.result.paragraphs if self.result.paragraphs else []

        # Parse the text blocks for figure and tables
        figure_block_indices = []
        table_block_indices = []
        figure_content = self.result.figures if self.result.figures else []
        table_content = self.result.tables if self.result.tables else []

        # Get the paragraph indices for figures and tables
        # Since one figure may contain multiple blocks, we need to maintain a mapping of paragraph index to figure index and table index
        figure_mapping = {}
        table_mapping = {}
        for figure_index,figure in enumerate(figure_content):
            figure_elements = figure.elements
            if figure.elements:
                sub_figure_block_indices = [int(element.split('/')[-1]) for element in figure_elements]
                figure_block_indices.extend(sub_figure_block_indices)
                # Add the mapping
                for block_index in sub_figure_block_indices:
                    figure_mapping[block_index] = figure_index
        for table_index,table in enumerate(table_content):
            if table.caption:
                sub_table_block_indices = [int(element.split('/')[-1]) for element in table.caption.elements]
                table_block_indices.extend(sub_table_block_indices)
                for block_index in sub_table_block_indices:
                    table_mapping[block_index] = table_index
            if table.cells:
                for cell in table.cells:
                    if cell.elements:
                        sub_cell_block_indices = [int(element.split('/')[-1]) for element in cell.elements]
                        table_block_indices.extend(sub_cell_block_indices)
                        for block_index in sub_cell_block_indices:
                            table_mapping[block_index] = table_index
            if table.footnotes:
                for footnote in table.footnotes:
                    if footnote.elements:
                        sub_footnote_block_indices = [int(element.split('/')[-1]) for element in footnote.elements]
                        table_block_indices.extend(sub_footnote_block_indices)
                        for block_index in sub_footnote_block_indices:
                            table_mapping[block_index] = table_index

        # Prepare the full text, exclude the paragraphs that are part of figures and tables
        full_text = ''
        registered_tables = []
        registered_figures = []
        last_page_num = 1
        for index, block in enumerate(text_blocks):
            if index not in figure_block_indices and index not in table_block_indices:
                current_page_num = block.bounding_regions[0].page_number # Page separation
                if current_page_num != last_page_num:
                    full_text = full_text.strip()
                    full_text += '\n\n\n'
                    last_page_num = current_page_num
                if block.role: # special text need further treatment
                    if block.role == 'sectionHeading':
                        if full_text[-2:] != '\n\n':
                            # Make sure heading is separated by two \n
                            full_text = full_text.strip() + '\n\n'
                        full_text += 'Section:' + block.content + '\n'
                    elif block.role == 'footnote':
                        # Remove the last \n from full_text, then add the footnote
                        full_text = full_text[:-1] + block.content + '\n\n'
                    elif block.role == 'title':
                        full_text += 'Title:' + block.content + '\n'
                    elif block.role == 'pageHeader':
                        # Skip this as it is useless
                        continue
                    elif block.role == 'pageFooter':
                        # Skip this as it is useless
                        continue
                    elif block.role == 'pageNumber':
                        # Skip this as it is useless
                        continue
                    else:
                        full_text += block.content + '\n\n'
                else:
                    if len(block.content.split(' ')) < 15:
                        # Avg sentence length is 15-20 words, if less than 15, consider it as incomplete text block
                        full_text += block.content + '\n'
                    else:
                        full_text += block.content + '\n\n'
            if index in table_block_indices and table_mapping[index] not in registered_tables:
                full_text += f'<Table_{table_mapping[index]}>\n'
                registered_tables.append(table_mapping[index])
            if index in figure_block_indices and figure_mapping[index] not in registered_figures:
                full_text += f'<Figure_{figure_mapping[index]}>\n'
                registered_figures.append(figure_mapping[index])

        # Replace processed figure text and table text to original place
        for figure_index,figure in enumerate(figure_content):
            figure_text = self.get_caption(figure)
            full_text = full_text.replace(f'<Figure_{figure_index}>\n',f'<Figure_{figure_index}>\n' + figure_text.strip() + '\n')
        for table_index,table in enumerate(table_content):
            table_text = self.get_formatted_table(table)
            full_text = full_text.replace(f'<Table_{table_index}>\n',f'<Table_{table_index}>\n' + table_text.strip() + '\n')
        self.full_text = full_text


    def get_image(self, page_num, polygen_list, dpi = azure_output_dpi):
        # Get the image from the pdf
        scale_factor = 72  # 1 inch = 72 points in PDF terms
        points = [polygen_list[0:2], polygen_list[2:4], polygen_list[4:6], polygen_list[6:8]]
        points.sort(key=lambda p: (p[1], p[0]))  # Sort by y first, then by x

        # Upper left point
        x1, y1 = points[0]

        # Down right point
        x2, y2 = points[-1]
        x1_pix = x1 * scale_factor
        y1_pix = y1 * scale_factor
        x2_pix = x2 * scale_factor
        y2_pix = y2 * scale_factor

        page = self.pdf_handle.load_page(page_num - 1)  # Page numbers are 0-indexed in PyMuPDF

        # Define the rectangle in pixels
        rect = fitz.Rect(x1_pix, y1_pix, x2_pix, y2_pix)

        # Take a screenshot of the specified region
        pix = page.get_pixmap(clip=rect, dpi=dpi)

        # Save the region to a img file
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        return img

    def combine_images(self, images):
        # Find the width of the widest image
        max_width = max(img.width for img in images)
        total_height = sum(img.height for img in images)

        # Create a new blank image with the appropriate size
        combined_image = Image.new("RGB", (max_width, total_height), (255, 255, 255))

        y_offset = 0
        for img in images:
            # Center align the images horizontally
            x_offset = (max_width - img.width) // 2
            combined_image.paste(img, (x_offset, y_offset))
            y_offset += img.height

        return combined_image

    def save_figure(self, base_save_dir):
        # Save the original image from the ocr layout result.
        # Use the inches as the unit. Read the bounding box of the figure and save the img in the corresponding page using fitz
        figure_content = self.result.figures if self.result.figures else []
        img_dir = os.path.join(base_save_dir, 'images')
        self._check_create_dir(img_dir)
        for figure_index,figure in enumerate(figure_content):
            sub_imgs = []
            page_num = figure.bounding_regions[0].page_number
            for bounding_region in figure.bounding_regions:
                # Get the image from the pdf
                sub_img = self.get_image(page_num, bounding_region.polygon)
                sub_imgs.append(sub_img)
            combined_img = self.combine_images(sub_imgs)
            # Rotate the image if the page is rotated.
            # Minus angle is counterclockwise in Azure definition
            # But it is the counter as PIL definition, so directly rotate is enough
            page_rotate_angle = self.result.pages[page_num - 1].angle
            combined_img = combined_img.rotate(page_rotate_angle, expand=True)
            output_filename = f"page{page_num}_img{figure_index}.png"
            combined_img.save(os.path.join(img_dir, output_filename))


    def save_table(self, base_save_dir):
        # Save the original image from the ocr layout result.
        # Use the inches as the unit. Read the bounding box of the figure and save the img in the corresponding page using fitz
        figure_content = self.result.tables if self.result.tables else []
        img_dir = os.path.join(base_save_dir, 'tables')
        self._check_create_dir(img_dir)
        for figure_index, figure in enumerate(figure_content):
            # Read the boundary region of the figure
            sub_imgs = []
            page_num = figure.bounding_regions[0].page_number
            for bounding_region in figure.bounding_regions:
                # Get the image from the pdf
                sub_img = self.get_image(page_num, bounding_region.polygon)
                sub_imgs.append(sub_img)
            combined_img = self.combine_images(sub_imgs)
            page_rotate_angle = self.result.pages[page_num - 1].angle
            combined_img = combined_img.rotate(page_rotate_angle, expand=True)
            output_filename = f"page{page_num}_table{figure_index}.png"
            combined_img.save(os.path.join(img_dir, output_filename))

    def save_json_result(self, base_save_dir):
        # Save the json result
        save_path = os.path.join(base_save_dir, 'azure_result.json')
        result_dict = self.result.as_dict()
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=4)

    def load_json_result(self, base_save_dir):
        from azure.ai.documentintelligence.models import AnalyzeResult
        # Load the json result
        save_path = os.path.join(base_save_dir, 'azure_result.json')
        with open(save_path, 'r', encoding='utf-8') as f:
            result_dict = json.load(f)
        self.result = AnalyzeResult(result_dict)

    def parse(self, paperpath, save_dir):
        # Reset the full text and text blocks
        self.full_text = ''
        self.text_blocks = None

        self.paperpath = paperpath
        self.save_dir = save_dir
        self.check_pdf_validaty()
        self.page_num = self.get_page_nums()
        self._check_create_dir(save_dir)

        # Call azure
        if self.auto_load_json and os.path.exists(os.path.join(save_dir, 'azure_result.json')):
            # First try to load the parse result json from save_dir
            try:
                print('Paper already parsed, trying to re-loading the json result')
                self.load_json_result(save_dir)
            except:
                warnings.warn('Failed to load the json result due to corruption, re-parsing the paper')
                self.poll_azure_result()
                self.save_json_result(save_dir)
        else:
            self.poll_azure_result()
            self.save_json_result(save_dir)

        # Text extraction
        self.text_parsing()
        self.save_txt(save_dir)

        # Load pdf handle
        self.pdf_handle = fitz.open(self.paperpath)
        # Figure extraction
        self.save_figure(save_dir)
        # Table extraction
        self.save_table(save_dir)
        self.pdf_handle.close()


class AzureMarkdownParser(AzureExtractor):
    def __init__(self):
        super().__init__()
        self.client = None
        self.result = None
        self.auto_load_json = None
        self.pdf_handle = None
        self.markdown_pages = []

    def initialize(self, endpoint, key, auto_load=True):
        from azure.core.credentials import AzureKeyCredential
        from azure.ai.documentintelligence import DocumentIntelligenceClient


        self.client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))
        self.auto_load_json = auto_load

    def poll_azure_result(self):
        with open(self.paperpath, "rb") as f:
            poller = self.client.begin_analyze_document(
                "prebuilt-layout",
                body=f,
                content_type="application/octet-stream",
                output_content_format="markdown"
            )
        self.result = poller.result()

    @staticmethod
    def convert_html_tables_to_csv(text):
        # Regular expression to match <table>...</table> blocks
        table_pattern = re.compile(r"<table>.*?</table>", re.DOTALL)

        # Find all tables in the text
        tables = table_pattern.findall(text)

        # Placeholder for the text with tables replaced by CSV strings
        updated_text = text

        for i, table_html in enumerate(tables):
            # Use pandas to read the HTML table
            table_buffer = StringIO(table_html)
            table_df = pd.read_html(table_buffer)[0]

            # Convert the DataFrame to a CSV-like string
            csv_buffer = StringIO()
            table_df.to_csv(csv_buffer, index=False, sep='|')
            csv_string = csv_buffer.getvalue()

            # Replace the HTML table with the CSV string in the text
            updated_text = updated_text.replace(table_html, f"<table>\n{csv_string.strip()}\n</table>")

        return updated_text

    @staticmethod
    def remove_figure_tags(text):
        # Regular expression to match <figure>...</figure> blocks
        figure_content_pattern = re.compile(r"(<figure>).*?(</figure>)", re.DOTALL)

        # Replace all matches with an empty string
        cleaned_text = re.sub(figure_content_pattern, r"\1\2", text)

        return cleaned_text

    @staticmethod
    def remove_page_controllers(text):
        # Regular expressions for page controllers
        footer_pattern = re.compile(r"<!--\s*PageFooter=.*?-->")
        header_pattern = re.compile(r"<!--\s*PageHeader=.*?-->")
        page_number_pattern = re.compile(r"<!--\s*PageNumber=.*?-->")

        # Remove each type of page controller
        text = re.sub(footer_pattern, "", text)
        text = re.sub(header_pattern, "", text)
        text = re.sub(page_number_pattern, "", text)

        # Return cleaned text
        return text

    @staticmethod
    def normalize_newlines(text):
        # Replace two or more consecutive newline characters with a single newline
        text = text.replace("\r", "")
        return re.sub(r'\n+', '\n', text)

    def save_markdown(self, base_save_dir):
        markdown_dir = os.path.join(base_save_dir, 'txt')
        if not os.path.exists(markdown_dir):
            os.makedirs(markdown_dir)

        markdown_path = os.path.join(markdown_dir, 'output.txt')
        markdown_text = self.result.content

        markdown_text = self.convert_html_tables_to_csv(markdown_text)
        markdown_text = self.remove_figure_tags(markdown_text)
        markdown_text = self.remove_page_controllers(markdown_text)
        markdown_text = self.normalize_newlines(markdown_text)

        with open(markdown_path, 'w', encoding='utf-8') as f:
            f.write(markdown_text)

    def parse(self, paperpath, save_dir):
        self.paperpath = paperpath
        self.save_dir = save_dir
        self.check_pdf_validaty()
        self.page_num = self.get_page_nums()
        self._check_create_dir(save_dir)

        # Call azure
        if self.auto_load_json and os.path.exists(os.path.join(save_dir, 'azure_markdown_result.json')):
            try:
                print('Paper already parsed, trying to re-loading the json result')
                self.load_json_result(save_dir)
            except:
                warnings.warn('Failed to load the json result due to corruption, re-parsing the paper')
                self.poll_azure_result()
                self.save_json_result(save_dir)
        else:
            self.poll_azure_result()
            self.save_json_result(save_dir)

        # Combine all markdown pages into full text
        self.save_markdown(self.save_dir)

        self.pdf_handle = fitz.open(self.paperpath)
        self.save_table(self.save_dir)
        self.save_figure(self.save_dir)


def paper_parse(paperpath, save_dir, extraction_engine: str):
    # Parse the paper using the specified engine
    global global_extractor
    if global_extractor is None:
        if extraction_engine == 'paddle':
            global_extractor = PaddleExtractor()
            global_extractor.initialize()
        elif extraction_engine == 'fitz':
            global_extractor = FitzExtractor()
            global_extractor.initialize()
        elif extraction_engine == 'nougat':
            global_extractor = NougatExtractor()
            global_extractor.initialize(parse_mode='cli', API_address = 'http://127.0.0.1:8503')
        elif extraction_engine == 'azure':
            global_extractor = AzureExtractor()
            global_extractor.initialize(azure_api_endpoint, azure_api_key, azure_auto_load)
        elif extraction_engine == 'azuremarkdown':
            global_extractor = AzureMarkdownParser()
            global_extractor.initialize(azure_api_endpoint, azure_api_key, azure_auto_load)
        else:
            raise Exception('Invalid extraction engine')

    global_extractor.parse(paperpath, save_dir)

if __name__ == '__main__':
    # Example of how to use these classes for batch processing
    root_dir = r'D:\data\OCM_articles_PDF'
    output_dir = r'D:\data\OCM_articles_MD'
    papers = os.listdir(root_dir)
    force_rerun = True
    paper_paths = [os.path.join(root_dir, paper) for paper in papers]
    for index, path in enumerate(paper_paths):
        save_dir = os.path.join(output_dir,papers[index].replace(".pdf",""))
        if os.path.exists(save_dir) and not force_rerun:
            continue
        s_time = time.time()
        try:
            paper_parse(path, save_dir, 'azuremarkdown')
        except Exception as e:
            print(f'Failed to extract paper {index}')
            traceback.print_exc()
        e_time = time.time()
        print(f'Paper {index} extracted in {e_time - s_time} seconds')