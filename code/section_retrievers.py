from edgar import *
import time
from edgar import set_identity
import re
import pandas as pd


class EdgarRetriever:
    def __init__(self, companies: List[str], years_back: int, credentials: str = None):
        self.companies = companies
        self.years_back = years_back
        self.sleep_time = 0.1

        if self._is_valid(credentials):
            set_identity(credentials)
            print("Identity successfully set")
        else:
            raise ValueError("Invalid credentials format, credentials must be formatted as 'firstname lastname email@domain.com'")
        
        self.section_patterns = {
            'risk factors': {
                'start': r"item 1a.\s*risk factors",
                'end': r"item 1b.\s*unresolved staff comments"
                },
            'managements discussion': {
                'start': r"item 7.\s*management",
                'end': r"item 7a.\s*(qualitative|quantitative)"
                }
        }

    @staticmethod
    def _clean_text(selected_raw_chunk_text: List[str]) -> list:
        for i in range(len(selected_raw_chunk_text)):
            selected_raw_chunk_text[i] = selected_raw_chunk_text[i].lower()
            selected_raw_chunk_text[i] = selected_raw_chunk_text[i].replace("\n"," ")
            selected_raw_chunk_text[i] = selected_raw_chunk_text[i].replace("•", " ")
            selected_raw_chunk_text[i] = selected_raw_chunk_text[i].replace("table of contents","")
            selected_raw_chunk_text[i] = " ".join(selected_raw_chunk_text[i].split())
        return selected_raw_chunk_text

    @staticmethod
    def _is_valid(credentials: str) -> bool:
        validpattern = r"[A-Za-z]+\s[A-Za-z]+\s[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
        return bool(re.match(validpattern, credentials))

    def get_filings(self):
        companies = []
        years = []
        sections = []
        text = []

        for company in self.companies:
            chunked_tenks = {}

            try:
                time.sleep(self.sleep_time)

                tenks = Company(company).get_filings(form="10-K").latest(self.years_back)

                if self.years_back == 1:  # deal with single year case, edgartools returns a single object
                    chunked_tenks[tenks.filing_date] = tenks.sections()
                else:
                    chunked_tenks = {tenks[i].filing_date: tenks[i].sections() for i in range((len(tenks)))}
                
                for year in chunked_tenks:
                    raw_chunked_text = chunked_tenks[year]

                    for desired_section in self.section_patterns:
                        start_pattern = self.section_patterns[desired_section]['start']
                        end_pattern = self.section_patterns[desired_section]['end']

                        index_start = None
                        index_end = None

                        for i in range(len(raw_chunked_text)):
                            if re.search(start_pattern, raw_chunked_text[i].lower()):
                                index_start = i
                            elif re.search(end_pattern, raw_chunked_text[i].lower()):
                                index_end = i
                                break

                        if index_start is not None and index_end is not None:
                            selected_raw_chunk_text = raw_chunked_text[index_start:index_end]
                            cleaned_chunk_text = self._clean_text(selected_raw_chunk_text)
                        else:
                            print("Could not find section: ", desired_section, " for company: ", company, " in year: ", year)
                        



            except Exception as e:
                print("Error retrieving filings for company: ", company)
        
        return None


if __name__ == '__main__':
    er = EdgarRetriever(["AAPL"], 1, "Neal Lockhart neal301@gmail.com")