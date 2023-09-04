import re
from bs4 import BeautifulSoup

def remove_whitespace(text:str) -> str:
    text = re.sub(r"[\n\t\s]+", " ", text) # remove tabs, newlines and spaces
    return text.strip()

class TextExtraction:
    def extract_text_from_html(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        title = self.extract_title(soup)
        description = self.extract_description(soup)
        keywords = self.extract_keywords(soup)
        body = self.extract_body(soup)
        text = " ".join([title, description, keywords, body])
        return remove_whitespace(text)

    def extract_title(self, soup: BeautifulSoup) -> str:
        title_element = soup.find("title")
        if title_element:
            title = title_element.get_text()
            if title:
                return remove_whitespace(title)
        return ""

    def extract_description(self, soup: BeautifulSoup) -> str:
        description_element = soup.find("meta", {"name": re.compile("(?i)description")})
        if description_element and description_element.get("content"):
            return remove_whitespace(description_element.get("content"))
        if description_element and description_element.get("value"):
            return remove_whitespace(description_element.get("value"))
        return ""

    def extract_keywords(self, soup: BeautifulSoup) -> str:
        keywords_element = soup.find("meta", {"name": re.compile("(?i)keywords")})
        if keywords_element and keywords_element.get("content"):
            return remove_whitespace(keywords_element.get("content"))
        if keywords_element and keywords_element.get("value"):
            return remove_whitespace(keywords_element.get("value"))
        return ""

    def extract_body(self, soup: BeautifulSoup) -> str:
        for element in soup(
            ["script", "style", "a", "head", "title", "meta", "footer", "form"]
        ):
            element.extract()
        return soup.get_text(strip=True, separator=" ")
