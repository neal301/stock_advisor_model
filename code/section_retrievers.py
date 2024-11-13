import List


def preprocess_text(chunked_text: List[str]) -> List[str]:
    cleaned_text = []
    
    for i in range(len(chunked_text)):
        chunked_text[i] = chunked_text[i].lower()
        chunked_text[i] = chunked_text[i].replace('\n', ' ')
        chunked_text[i] = chunked_text[i].replace('•', ' ')
        chunked_text[i] = chunked_text[i].replace("table of contents","")
        chunked_text[i] = chunked_text[i].strip()
        cleaned_text.append(chunked_text[i])
    
    return cleaned_text

def 