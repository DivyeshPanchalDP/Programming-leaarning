import pytesseract
from PIL import Image
img=Image.open(r"D:\Programming leaarning\covid 19\ai\example-page-001.jpg")
text=pytesseract.image_to_string(img)
print(text)