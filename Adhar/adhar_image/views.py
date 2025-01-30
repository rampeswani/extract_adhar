from django.shortcuts import render
# import pytesseract 
import cv2

# cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
# print("kk - cascade_path",cascade_path)
# # face_cascade = cv2.CascadeClassifier("D://haarcascade_frontalface_default.xml")
# face_cascade = cv2.CascadeClassifier("static/haarcascade_frontalface_default.xml")

# if face_cascade.empty():
#     print("Error: Haar cascade file not found!")
# else :
#     print("loaded",face_cascade)
import re
from django.http import JsonResponse
from PIL import Image
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
import os
import base64


# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Create your views here.

    # image = cv2.imread(image_path)
    # height, width, _ = image.shape

    # # Split the image into left and right parts
    # left_image = image[:, :width // 2]  # Left half of the image
    # right_image = image[:, width // 2:]  # Right half of the image

    # # Perform OCR on both left and right parts
    # left_text = pytesseract.image_to_string(left_image, lang='eng+hin')
    # right_text = pytesseract.image_to_string(right_image, lang='eng+hin')

    # # Combine the extracted texts from both parts
    # full_text = left_text + "\n" + right_text
    # print("Extracted Text: ", full_text)

    # return full_text

def extract_text_from_image(image_path):
    reader = easyocr.Reader(['en', 'hi'])  # You can include more languages if necessary
    result = reader.readtext(image_path, detail=0)  # detail=0 returns only the text

    # Join the results into a single string of text
    full_text = " ".join(result)
    print("Extracted Text: ", full_text)

    return full_text





def extract_aadhaar_details(ocr_text,image_path):
    cleaned_text = " ".join(ocr_text.split())
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        image_data = f"data:image/jpeg;base64,{encoded_image}"
    
    if "Aadhar" in ocr_text or "adhar" in ocr_text or "आधार" in ocr_text:


        print("yes",ocr_text)

    if "Driving License" in ocr_text :
        extract_driving_license_details(cleaned_text,image_data)
    
    list  = cleaned_text.split()
    
    print("cleaned_text",cleaned_text)
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        image_data = f"data:image/jpeg;base64,{encoded_image}"


    name_pattern = r"([A-Z][a-z]+(?: [A-Z][a-z]+)+)(?=\sSO:)"  # Matches one or more capitalized words before 'SO:'


    father_name_pattern = r"SO:\s([A-Za-z\s]+)"  # Matches Father's Name after 'SO:'
    father_name_pattern_2 = r"sJo:\s([A-Za-z\s]+)"  # Matches Father's Name after 'SO:'
    dob_pattern = r"तिथि/D08:\s(\d{2}/\d{2}/\d{4})"  # Matches DOB (e.g., '08/10/2002')
    dob_2  = r"/D0B:\s*(\d{2}/\d{2}/\d{4})"
    aadhaar_pattern = r"\b(\d{4})\s?(\d{4})\s?(\d{4})\b"  # Matches Aadhaar number '4704 1229 4420' strictly



    # Extract Name
    # Extract Name (after "Government of India पेसवानी")
    name = "Not Found"
    name_match = re.search(r"Government of India\s+पेसवानी\s+([A-Z][a-z]+ [A-Z][a-z]+)", cleaned_text)
    if name_match :
        name = name_match.group(1).strip()
    
    

    word_list = cleaned_text.split()

    # Extract Name (index 22 and 23 based on your list)
    
    if name == 'Not Found':
        name = word_list[21] + " " + word_list[22]  # Combining 'Ram' and 'Peswani'


    # Extract Father's Name
    father_name = "Not Found"
    father_name_match = re.search(father_name_pattern, cleaned_text)
    if father_name_match:
        father_name = father_name_match.group(1).strip()

    if father_name == 'Not Found':
        father_name_match = re.search(father_name_pattern_2, cleaned_text)
        if father_name_match :
            father_name = father_name_match.group(1).strip()
        

    # Extract DOB (in Hindi format as shown in your example)
    dob = "Not Found"
    dob_match = re.search(dob_pattern, cleaned_text)
    if dob_match:
        dob = dob_match.group(1).strip()
    if dob == "Not Found" :
        dob_match = re.search(dob_2, cleaned_text)
        if(dob_match):
            dob = dob_match.group(1).strip()
    # Extract Aadhaar number
    address_pattern = r"(\d{6})"  # This matches a 6-digit pincode pattern, likely to be part of the address
    cleaned_text_without_address = re.sub(address_pattern, "", cleaned_text)

    # **Step 2: Extract Aadhaar number from the cleaned text (without address)**
    aadhaar = "Not Found"
    aadhaar_matches = re.findall(aadhaar_pattern, cleaned_text_without_address)
    if len(aadhaar_matches) > 0:
        # Join the matched groups into the final Aadhaar number
        aadhaar = ''.join(aadhaar_matches[0])  # e.g., '470412294420'

    # Return details as a dictionary
    return {
        "Name": name,
        "Father's Name": father_name,
        "DOB": dob,
        # "Issue Date": issue_date,
        "Aadhaar Number": aadhaar,
        
        "Image": image_data  
    }





def extract_adhar_details(cleaned_text,image_data,image_path):
    name_pattern = r"([A-Z][a-z]+(?: [A-Z][a-z]+)+)(?=\sSO:)"  # Matches one or more capitalized words before 'SO:'


    father_name_pattern = r"SO:\s([A-Za-z\s]+)"  # Matches Father's Name after 'SO:'
    father_name_pattern_2 = r"sJo:\s([A-Za-z\s]+)"  # Matches Father's Name after 'SO:'
    dob_pattern = r"तिथि/D08:\s(\d{2}/\d{2}/\d{4})"  # Matches DOB (e.g., '08/10/2002')
    dob_2  = r"/D0B:\s*(\d{2}/\d{2}/\d{4})"
    aadhaar_pattern = r"\b(\d{4})\s?(\d{4})\s?(\d{4})\b"  # Matches Aadhaar number '4704 1229 4420' strictly



    # Extract Name
    # Extract Name (after "Government of India पेसवानी")
    name = "Not Found"
    name_match = re.search(r"Government of India\s+पेसवानी\s+([A-Z][a-z]+ [A-Z][a-z]+)", cleaned_text)
    if name_match :
        name = name_match.group(1).strip()
    
    

    word_list = cleaned_text.split()

    # Extract Name (index 22 and 23 based on your list)
    
    if name == 'Not Found':
        name = word_list[21] + " " + word_list[22]  # Combining 'Ram' and 'Peswani'


    # Extract Father's Name
    father_name = "Not Found"
    father_name_match = re.search(father_name_pattern, cleaned_text)
    if father_name_match:
        father_name = father_name_match.group(1).strip()

    if father_name == 'Not Found':
        father_name_match = re.search(father_name_pattern_2, cleaned_text)
        if father_name_match :
            father_name = father_name_match.group(1).strip()
        

    # Extract DOB (in Hindi format as shown in your example)
    dob = "Not Found"
    dob_match = re.search(dob_pattern, cleaned_text)
    if dob_match:
        dob = dob_match.group(1).strip()
    if dob == "Not Found" :
        dob_match = re.search(dob_2, cleaned_text)
        if(dob_match):
            dob = dob_match.group(1).strip()
    # Extract Aadhaar number
    address_pattern = r"(\d{6})"  # This matches a 6-digit pincode pattern, likely to be part of the address
    cleaned_text_without_address = re.sub(address_pattern, "", cleaned_text)

    # **Step 2: Extract Aadhaar number from the cleaned text (without address)**
    aadhaar = "Not Found"
    aadhaar_matches = re.findall(aadhaar_pattern, cleaned_text_without_address)
    if len(aadhaar_matches) > 0:
        # Join the matched groups into the final Aadhaar number
        aadhaar = ''.join(aadhaar_matches[0])  # e.g., '470412294420'

    person_photo = extract_face(image_path)

    # Return details as a dictionary
    return {
        "Name": name,
        "Father's Name": father_name,
        "DOB": dob,
        # "Issue Date": issue_date,
        "Aadhaar Number": aadhaar,
        
        "Image": person_photo  
    }




def extract_driving_license_details(cleaned_text, image_data,image_path):
    # Driving License Patterns
    
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)
    print("text inside the license block " , cleaned_text )

    # Improved Patterns
    name_pattern = r"Name\s*[:\-]?\s*([A-Z]+\s[A-Z]+)"

    father_name_pattern = r"[S|W|D][./\s]*of\s*[:\-]?\s*([A-Z]+\s[A-Z]+)"

    dob_pattern = r"(?:DOB|Date of Birth)\s*[:\-]?\s*(\d{2}[-/]\d{2}[-/]\d{4})"

    license_no_pattern = r"\b([A-Z]{2}\d{2}[A-Z]-\d{4}-\d{7})\b"
    # license_no_pattern = r"\b([A-Z]{2}\d{2}[A-Z]-\d{4}-\d{7})\b"
    name_match = re.search(name_pattern, cleaned_text)
    name = name_match.group(1).strip() if name_match else "Not Found"

    # Extract Father's Name
    father_name_match = re.search(father_name_pattern, cleaned_text)
    father_name = father_name_match.group(1).strip() if father_name_match else "Not Found"

    # Extract DOB (Handling both "08-10-2002" and "08/10/2002")
    dob_match = re.search(dob_pattern, cleaned_text)
    dob = dob_match.group(1).strip() if dob_match else "Not Found"

    # Extract License Number
    license_no = re.search(license_no_pattern, cleaned_text)
    license_no = license_no.group(1).strip() if license_no else "Not Found"

    person_photo = extract_face(image_path)


    return {
        "Document Type": "Driving License",
        "Name": name,
        "Father's Name": father_name,
        "DOB": dob,
        "License Number": license_no,
        "Image": image_data,
        "person_photo" : person_photo
    }



def extract_details_from_doc(ocr_text,image_path):
    cleaned_text = " ".join(ocr_text.split())
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        image_data = f"data:image/jpeg;base64,{encoded_image}"
    if "Driving License" in ocr_text :
        return extract_driving_license_details(cleaned_text,image_data,image_path)

    else :
        return extract_adhar_details(cleaned_text,image_data,image_path)


# def extract_face(image_path):


#     cascade_path = r'C:\Users\Newsainathcomputer\AppData\Local\Programs\Python\Python310\lib\site-packages\cv2\data\haarcascade_frontalface_default.xml'
    
#     # Load the cascade
#     face_cascade = cv2.CascadeClassifier(cascade_path)

#     if face_cascade.empty():
#         return "Error: Haar cascade file not found!"

#     # Load the image
#     image = cv2.imread(image_path)
#     if image is None:
#         return f"Error: Could not read image at {image_path}"

#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Detect faces
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     if len(faces) == 0:
#         return "No Face Detected"

#     # Extract the first face found
#     x, y, w, h = faces[0]
#     face_crop = image[y:y+h, x:x+w]

#     # Encode the cropped face image to Base64
#     _, buffer = cv2.imencode('.jpg', face_crop)
#     encoded_face = base64.b64encode(buffer).decode('utf-8')

#     return f"data:image/jpeg;base64,{encoded_face}"
def extract_face(image_path):
    import cv2
    import base64
    import os

    # Get Haarcascade path dynamically
    # cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    
    # Load the cascade
    face_cascade = cv2.CascadeClassifier("static/haarcascade_frontalface_default.xml")
    # face_cascade = cv2.CascadeClassifier("D://haarcascade_frontalface_default.xml")

    if face_cascade.empty():
        return "Error: Haar cascade file not found!"

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        return f"Error: Could not read image at {image_path}"

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return "No Face Detected"

    # Extract the first face found
    x, y, w, h = faces[0]
    face_crop = image[y:y+h, x:x+w]

    # Encode the cropped face image to Base64
    _, buffer = cv2.imencode('.jpg', face_crop)
    encoded_face = base64.b64encode(buffer).decode('utf-8')

    return f"data:image/jpeg;base64,{encoded_face}"

@csrf_exempt
def process_aadhaar_image(request):
    """
    API to process Aadhaar card image and extract details.
    """
    if request.method == 'POST' and request.FILES.get('image'):
        
        uploaded_file = request.FILES['image']
        file_path = default_storage.save(uploaded_file.name, uploaded_file)

        try:
        
            ocr_text = extract_text_from_image(file_path)
            details = extract_details_from_doc(ocr_text,file_path)
            response = {"status": "success", "data": details}
        except Exception as e:
            response = {"status": "error", "message": str(e)}
        finally:
        
            if os.path.exists(file_path):
                os.remove(file_path)

        return JsonResponse(response)

    return JsonResponse({"status": "error", "message": "Invalid request"}, status=400)



# def OCR(image_path):
#     apikey=K86700713388957
#     path = image_path 
#     https://api.ocr.space/parse/imageurl?apikey=helloworld&url=https://dl.a9t9.com/ocr/solarcell.jpg

import json 
import json
import requests
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def OCR(request):
    if request.method == "POST":
        try:
            import os
            import json
            import requests
            from django.http import JsonResponse

            # Parse the file path from the request body
            data = json.loads(request.body)
            file_path = data.get("file_path")

            if not file_path or not os.path.exists(file_path):
                return JsonResponse({"error": "Invalid or missing file path"}, status=400)

            # OCR API details
            api_key = "K86700713388957"
            ocr_url = "https://api.ocr.space/parse/image"

            # Extract the file extension
            _, file_extension = os.path.splitext(file_path)
            file_type = file_extension.lstrip(".")  # Remove the dot (e.g., "jpg")

            # Open the file in binary mode
            with open(file_path, "rb") as file:
                # Prepare the request
                response = requests.post(
                    ocr_url,
                    files={"file": file},
                    data={"apikey": api_key, "filetype": file_type},
                )


            with open(file_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        
                image_data = f"data:image/jpeg;base64,{encoded_image}"

            # Check for successful response
            if response.status_code == 200:
                ocr_data = response.json()
                parsed_text = ocr_data.get("ParsedResults", [{}])[0].get("ParsedText", "")
                print("parsed text  == ",parsed_text)

                # Extract information using regular expressions
                #name = re.search(r"(?<!S/)OHAAR\s+(.*?)\s", parsed_text)
                #//father_name = re.search(r"S/O:\s*(.*?)\s*,", parsed_text)
                aadhar_number = re.search(r"(\d{4} \d{4} \d{4})", parsed_text)
                dob = re.search(r"DOB:\s*([\d/]+)", parsed_text)

                name_pattern = r"OHAAR\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)"
                father_name_pattern = r"S/O:\s*([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)"
                name="not found"
                name_matches = re.findall(name_pattern, parsed_text)
                if len(name_matches) > 0:
                    name = name_matches[0].strip().replace("'", "")  

                father_name_match = re.findall(father_name_pattern, parsed_text)
                if len(father_name_match) > 0 :
                    father_name = father_name_match[0].strip().replace("'","")
                # Create the result dictionary
                result = {
                    # "Name": name.group(1).strip() if name else None,
                    "Name" : name,
                    # "FatherName": father_name.group(1).strip() if father_name else None,
                    "Father's Name" : father_name ,
                    "AadharNumber": aadhar_number.group(1) if aadhar_number else None,
                    "DOB": dob.group(1) if dob else None,
                    "Image" : image_data
                }



                return JsonResponse(result)
            else:
                return JsonResponse(
                    {"error": "OCR API call failed", "details": response.text},
                    status=response.status_code,
                )

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON input"}, status=400)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    else:
        return JsonResponse({"error": "Invalid HTTP method"}, status=405)







# import textract
# image_path = "D:\downloads_new\Capture.JPG"
# print(image_path)
# text = textract.process(image_path)
# print("xtracted text :")
# print(text.decode('utf-8'))


# from transformers import TrOCRProcessor, VisionEncoderDecoderModel
# from PIL import Image

# # Load the model and processor
# processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
# model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# # Load your image
# image = Image.open("D:\downloads_new\Capture.JPG").convert("RGB")

# # Preprocess and predict
# pixel_values = processor(image, return_tensors="pt").pixel_values
# generated_ids = model.generate(pixel_values)
# text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# print(f"Detected text: {text}")


import easyocr

# # Initialize the EasyOCR reader (you can specify languages here if needed)
# reader = easyocr.Reader(['en'])  # 'en' stands for English language

# # Path to the image file
# image_path = 'D:\downloads_new\CCapture.JPG'

# # Use the reader to extract text from the image
# result = reader.readtext(image_path)

# # Print the extracted text
# for detection in result:
#     print(detection[1])  # The second element contains the extracted text
