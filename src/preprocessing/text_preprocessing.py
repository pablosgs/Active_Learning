import re


class TextPreprocessor:
    """
    Preprocesses text for the content classification task
    """

    def preprocess_data(self, text: str) -> str:
        text = self.to_lowercase(text)
        text = self.remove_numerical_words(text)
        text = self.remove_special_characters(text)
        text = self.remove_excessive_whitespace(text)
        return text

    def to_lowercase(self, text: str) -> str:
        # Transform all uppercase letters to lowercase
        return text.lower()

    def remove_numerical_words(self, text: str) -> str:
        # Remove words with numerical characters
        text = re.sub(r"\S*\d\S*", "", text)
        return text

    def remove_special_characters(self, text: str) -> str:
        # Remove special characters - excluding "/", "-", and "."
        text = re.sub(r"[^a-z\/\-\. ]", "", text)
        text = re.sub(" +", " ", text)
        return text

    def remove_excessive_whitespace(self, text: str) -> str:
        # Remove newlines, tabs and excessive whitespace
        text = re.sub(r"(\n)|(\t)", "", text)
        text = re.sub(" +", " ", text)
        return text.strip()
