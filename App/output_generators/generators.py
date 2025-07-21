import json
from datetime import datetime
from fpdf import FPDF

def generate_json(data):
    """
    Creates a JSON file of the fields extracted from the transcript.

    Args:
        data (BaseModel): Extracted fields from the transcript (Pydantic model).

    Returns:
        dict: JSON payload of the extracted fields.
    """
    payload = {
        "case_id": f"CS-{datetime.now().strftime('%Y%m%d')}-0001",
        "customer": {
            "name": data.customer_name,
            "contact": data.contact_info
        },
        "order": {
            "number": data.order_number,
            "product": data.product_name,
            "purchase_date": data.date_of_purchase
        },
        "issue": {
            "description": data.issue_description,
            "preferred_resolution": data.preferred_resolution
        },
        "source": "transcript_parser",
        "created_at": datetime.now().isoformat()
    }

    with open("outputs/json_format.json", "w") as f:
        json.dump(payload, f, indent=2)
        
    return payload


def generate_pdf(data):
    """
    Creates a PDF file of the fields extracted from the transcript.

    Args:
        data (BaseModel): Extracted fields from the transcript (Pydantic model).

    Returns:
        None
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.set_title("Customer Service Summary")
    
    data_dict = data.dict()

    for key, value in data_dict.items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)

    pdf.output("outputs/pdf_summary.pdf")
