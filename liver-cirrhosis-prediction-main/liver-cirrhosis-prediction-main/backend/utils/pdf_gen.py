from fpdf import FPDF
import datetime

class DiagnosisPDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'LiverAI Smart Healthcare Assistant Report', 0, 1, 'C')
        self.ln(10)

def generate_medical_pdf(patient_name, stage, confidence, symptoms, report_id):
    pdf = DiagnosisPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.cell(200, 10, f"Report ID: {report_id}", ln=True)
    pdf.cell(200, 10, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.cell(200, 10, f"Patient Name: {patient_name}", ln=True)
    pdf.ln(5)
    
    pdf.set_text_color(255, 0, 0)
    pdf.cell(200, 10, f"Final Diagnosis: {stage}", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(200, 10, f"AI Confidence Score: {confidence}", ln=True)
    
    pdf.ln(10)
    pdf.cell(200, 10, "Analyzed Symptoms:", ln=True)
    for key, value in symptoms.items():
        pdf.cell(200, 8, f"- {key}: {value}", ln=True)
    
    pdf.ln(10)
    pdf.multi_cell(0, 10, "Note: This is an AI-generated screening. Please consult a human hepatologist for final confirmation.")
    
    path = f"reports/report_{report_id}.pdf"
    pdf.output(path)
    return path