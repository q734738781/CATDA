import fitz  # PyMuPDF

def is_image_based_pdf(pdf_path, threshold = 100):
    '''
    Utility function to determine if a PDF is image-based, or pymupdf supports the font in pdf file
    Compare a length of the text greater than a thresold and "\xef\xbf\xbd"
    '''
    doc = fitz.open(pdf_path)
    text_length = 0
    unsupported_chars = 0
    for page in doc:
        page_text = page.get_text()
        unsupported_chars += page_text.count('�')
        text_length += (len(page_text) - unsupported_chars)

    # Check if the text length is less than the threshold
    if text_length < threshold:
        return True  # The PDF is likely image-based

    # Check if the text contains the replacement character (indicating an unsupported font)
    if unsupported_chars > text_length:
        return True  # The PDF likely contains an unsupported font

    return False



