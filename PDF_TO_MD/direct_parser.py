import fitz  # PyMuPDF

def extract_text_blocks_with_fitz(pdf_path):
    '''
    Utility function to extract text blocks using Fitz.
    '''
    text_blocks = []
    doc = fitz.open(pdf_path)
    for page in doc:
        blocks = page.get_text('blocks')
        # Blocks are in the format:
        # (x0, y0, x1, y1, block_text, block no, block type)
        # block_type is 1 for an image block, 0 for text. block_no is the block sequence number
        text_blocks.append(blocks)
    doc.close()
    return text_blocks
