import fitz

def extract_pdf_link(pdf_path, page_index):

    # Creating a document object
    doc = fitz.open(pdf_path)

    # Extract number of pages
    page_count = doc.page_count
    print(f"Number of pages: {page_count}")

    # Get specified page by index
    page = doc.load_page(page_index)

    # text Read from pages
    page_text = page.get_text()
    print(f"\nPage {page_index + 1} Text:\n{page_text}")

    # Get all links from the page
    page_links = [link['uri'] for link in page.get_links()]
    print(f"\nPage {page_index + 1} Links:\n{page_links}")

    return page_count, page_text, page_links

# Specify path to the PDF file and the index of the page

pdf_file_path = "Machine_Learning_Task (3).pdf"
page_index_to_process = 1

# Call the function
num_pages, page_text, page_links = extract_pdf_link(pdf_file_path, page_index_to_process)



import requests
from io import BytesIO
from PIL import Image
import os

def download_images_from_links(links, output_folder="images"):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    for index, link in enumerate(links):
        try:
            response = requests.get(link)
            if response.status_code == 200:
                # Read the image from the response content
                image_data = BytesIO(response.content)
                image = Image.open(image_data)

                # Save the image to the output folder
                image_path = os.path.join(output_folder, f"image_{index + 1}.jpg")
                image.save(image_path)

                print(f"Image {index + 1} downloaded and saved at: {image_path}")
            else:
                print(f"Failed to download image from link: {link}")
        except Exception as e:
            print(f"Error downloading image from link {link}: {str(e)}")

# Call the function with the extracted links
download_images_from_links(page_links)
