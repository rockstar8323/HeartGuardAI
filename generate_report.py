from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_JUSTIFY

def create_pdf(filename="HeartGuardAI_Project_Report.pdf"):
    doc = SimpleDocTemplate(filename, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)
    
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY, fontSize=11, leading=14))
    styles.add(ParagraphStyle(name='SubHeading', fontSize=14, leading=16, spaceAfter=8, textColor='#2c3e50', fontName='Helvetica-Bold'))
    
    Story = []
    
    title_style = styles['Heading1']
    title_style.alignment = 1
    
    Story.append(Paragraph("HeartGuard AI: Full-Stack Heart Disease Predictor", title_style))
    Story.append(Spacer(1, 12))
    Story.append(Paragraph("Project Architecture & Implementation Status", styles['Heading3']))
    Story.append(Spacer(1, 24))

    # --- PART 1: FRONTEND ---
    Story.append(Paragraph("Part 1: Frontend Development", styles['Heading2']))
    Story.append(Spacer(1, 8))
    frontend_text = """
    <b>Technologies:</b> HTML5, CSS3, JavaScript.<br/><br/>
    <b>Current Status & Features:</b><br/>
    We have built a responsive, modern user interface designed to feel clinical yet accessible. The frontend provides a secure user authentication system (login/signup) leading to a centralized dashboard. The core of the frontend is the dual-mode prediction system:<br/>
    • <b>Quick Check Mode:</b> A simplified form utilizing 7 easy-to-answer questions for a rapid preliminary assessment.<br/>
    • <b>Detailed Clinical Mode:</b> A comprehensive 13-parameter form. It features a drag-and-drop file upload zone allowing users to upload medical reports (images or PDFs) directly into the browser, and the frontend communicates asynchronously via the Fetch API to parse this data and auto-fill the form.<br/>
    • <b>Results Presentation:</b> After prediction, the UI displays dynamic, visually distinct risk assessments alongside dynamic embedded charts.
    """
    Story.append(Paragraph(frontend_text, styles['Justify']))
    Story.append(Spacer(1, 16))

    # --- PART 2: BACKEND & OCR ---
    Story.append(Paragraph("Part 2: Backend & OCR Integration", styles['Heading2']))
    Story.append(Spacer(1, 8))
    backend_text = """
    <b>Technologies:</b> Python, Flask, SQLite, PyTesseract, PDFPlumber.<br/><br/>
    <b>Current Status & Features:</b><br/>
    The Flask backend orchestrates the entire application. It handles routing, secure user sessions via Flask-Login, and persists all user predictions into an SQLite database for history tracking.<br/>
    • <b>Prediction Endpoints:</b> We have modular REST-like endpoints serving predictions.<br/>
    • <b>Optical Character Recognition (OCR):</b> We implemented a <font name="Courier">report_parser.py</font> module capable of handling both PDFs (via pdfplumber) and images (via PyTesseract). We utilize advanced Regular Expressions (Regex) to scan the extracted text for clinical markers (e.g., "cholesterol", "max heart rate", "mmHg") and convert them into structured numerical data. <i>(Note: A fallback mechanism returning sample values is currently in place for environments where the native Tesseract C++ engine cannot be installed.)</i>
    """
    Story.append(Paragraph(backend_text, styles['Justify']))
    Story.append(Spacer(1, 16))

    # --- PART 3: ML + SHAP ---
    Story.append(Paragraph("Part 3: Machine Learning Model & SHAP Interpretability", styles['Heading2']))
    Story.append(Spacer(1, 8))
    ml_text = """
    <b>Dataset:</b> The models were trained using the renowned UCI Heart Disease Dataset containing 14 operational columns (including the severity target).<br/><br/>
    <b>Current Status & Features:</b><br/>
    • <b>Models:</b> We deployed an ensemble approach, relying on a Logistic Regression backbone to achieve an accuracy of approximately 88.5%. We separated our models into two artifacts: the <b>Detailed Model (13 features)</b> and a highly optimized <b>Quick Model (7 features)</b>, each serialized via joblib with their own standard scalers.<br/>
    • <b>Explainable AI (SHAP):</b> To prevent the ML from acting as a "black box", we integrated SHAP (SHapley Additive exPlanations). For every prediction, the <font name="Courier">shap.LinearExplainer</font> calculates the exact positive or negative impact of every single feature. The backend dynamically draws a horizontal bar chart and encodes it as a base64 string directly into our frontend HTML, explaining exactly *why* a specific prediction was made.<br/><br/>
    
    <b>Future Improvements for the Model:</b><br/>
    1. <b>More Extensive Datasets:</b> Acquire a larger, more globally diverse dataset to prevent demographic overfitting.<br/>
    2. <b>Advanced Architectures:</b> Migrate from standard Logistic Regression/Ensembles to non-linear tree-based models like XGBoost or a Deep Neural Network to catch nuanced correlations.<br/>
    3. <b>Vision-Language Models for OCR:</b> Replace regex-based parsing with local LLMs (like LLaVA) or Gemini/GPT-4V APIs to reliably extract data from highly unstructured, handwritten doctor prescriptions.<br/>
    4. <b>Holistic Parameters:</b> Add lifestyle and genetic features (dieting habits, family history of strokes) to improve true clinical accuracy.
    """
    Story.append(Paragraph(ml_text, styles['Justify']))

    doc.build(Story)

if __name__ == "__main__":
    create_pdf()
    print("PDF generated successfully.")
