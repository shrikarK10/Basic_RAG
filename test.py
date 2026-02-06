from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

try:
    print("Sending request...")
    response = client.models.generate_content(
        model="gemini-2.0-flash", 
        contents="hi"
    )
    print("Success! Response:")
    print(response.text)

except Exception as e:
    print(f"Error: {e}")