import requests

def translate_text(text, target_language, source_language='hi', api_key='your-api-key'):
    url = "https://api.mymemory.translated.net/get"  # MyMemory API endpoint
    params = {
        'q': text,                       # The text to translate
        'langpair': f"{source_language}|{target_language}",  # Source|Target language pair
        'key': api_key                   
    }
    response = requests.get(url, params=params)  # Sending GET request with params
    return response.json()['responseData']['translatedText']  # Extracting translated text from JSON response

api_key = '266f0c55837af3a24e12' 
translated_text = translate_text("यह एक उदाहरण वाक्य है।", 'en', api_key=api_key)
print("Translated Text:", translated_text)


