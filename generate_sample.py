from PIL import Image, ImageDraw, ImageFont

text_content = """CARDIOLOGY CLINIC - PATIENT MEDICAL REPORT

Patient Information:
Patient Name: John Doe
Age: 55 years old
Sex: Male

Vitals & Blood Work:
Resting Blood Pressure: 135 mmHg
Total Cholesterol: 245 mg/dl
Fasting Blood Sugar: 110 mg/dl

Stress Test & ECG Results:
Resting ECG: Normal
Max Heart Rate: 154 bpm
Exercise Induced Angina: No
ST Depression: 1.2 mm
ST Slope: flat

Angiography & Additional Info:
Major vessels: 0 blocked
Thalassemia: normal
"""

img = Image.new('RGB', (600, 600), color=(255, 255, 255))
d = ImageDraw.Draw(img)

try:
    font = ImageFont.truetype("arial.ttf", 20)
except IOError:
    font = ImageFont.load_default()

d.text((50, 50), text_content, fill=(0, 0, 0), font=font)

img.save('sample_medical_report.png')
print("Successfully generated sample_medical_report.png")
