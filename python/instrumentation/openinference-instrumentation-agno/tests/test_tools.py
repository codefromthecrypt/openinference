import requests
from agno.tools import Toolkit
from bs4 import BeautifulSoup


class SinglePageWebsiteTools(Toolkit):
    """WebsiteTools doesn't handle redirects nicely so rolled own."""

    def __init__(self) -> None:
        super().__init__(name="simple_website_tools")
        self.register(self.read_url)

    def read_url(self, url: str) -> str:
        response = requests.get(url, allow_redirects=True, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text(separator=" ", strip=True)
