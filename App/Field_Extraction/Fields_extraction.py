from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Defines the structure for a customer support case using Pydantic
class CustomerSupportCase(BaseModel):
    customer_name: str = Field(description="Full name of the customer")
    contact_info: str = Field(description="Email address or phone number of the customer")
    order_number: str = Field(description="Order number related to the issue")
    product_name: str = Field(description="Name of the product involved in the issue")
    date_of_purchase: str = Field(description="Date when the product was purchased")
    issue_description: str = Field(description="Detailed description of the issue faced by the customer")
    preferred_resolution: str = Field(description="Customer's preferred resolution such as refund, replacement, etc.")

def extract_fields():
    """
    Creates a LangChain pipeline to extract fields from a transcript using a prompt,
    a Gemini model, and a Pydantic parser.

    Returns:
        chain: A chain combining the prompt, model, and parser.
    """
    load_dotenv()
    
    parser = PydanticOutputParser(pydantic_object=CustomerSupportCase)

    prompt = PromptTemplate(
        template="""
            Extract the following fields from the customer service transcript:

            Fields to extract:
            - Customer Name
            - Contact Info (email or phone)
            - Order Number
            - Product Name
            - Date of Purchase (even if customer says "received on" or "got it on", infer this as the date of purchase)
            - Issue Description
            - Preferred Resolution

            Always infer the most accurate values, even if indirectly stated. If a field is truly not provided, write "Not provided in transcript".

            Return the output as valid JSON in the exact format.

            Transcript:
            {transcript}

            {format_instructions}
        """,
        input_variables=['transcript'],
        partial_variables={'format_instructions': parser.get_format_instructions()}
    )

    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    
    chain = prompt | model | parser
    return chain
