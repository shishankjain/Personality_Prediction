from pyresparser import ResumeParser

data = ResumeParser('SHISHANK JAIN- RESUME.pdf').get_extracted_data()
print (data)