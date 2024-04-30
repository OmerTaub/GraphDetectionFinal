import PyPDF2
import pandas as pd

# Open the PDF file
pdfFile = open('meetingminutes.pdf', 'rb')

pdfReader = PyPDF2.PdfFileReader(pdfFile)
pdfReader.numPages

# Extract the text from the first page
pageObj = pdfReader.getPage(0)
pageObj.extractText()

# Loop through all the pages
for pageNum in range(pdfReader.numPages):
    print(pdfReader.getPage(pageNum).extractText())
    # create a DataFrame to store the text
    df = pd.DataFrame(pdfReader.getPage(pageNum).extractText())
    print(df)

