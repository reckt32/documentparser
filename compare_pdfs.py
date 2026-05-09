import fitz

def extract_page_image(pdf_path, page_num, output_path):
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num - 1)  # 0-indexed
    pix = page.get_pixmap(dpi=150)
    pix.save(output_path)
    print(f"Saved {output_path}")

extract_page_image("test_report_v2.pdf", 7, "test_p7.png")
