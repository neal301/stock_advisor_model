from edgar import *
import time
from edgar import set_identity
import re


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
    def _is_valid(credentials: str) -> bool:
        validpattern = r"[A-Za-z]+\s[A-Za-z]+\s[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
        return bool(re.match(validpattern, credentials))

    def get_filings(self):
        filings = {}

        for company in self.companies:
            try:
                time.sleep(self.sleep_time)

                tenks = Company(company).get_filings(form="10-K").latest(self.years_back)

                if self.years_back == 1:  # deal with single year case, edgartools returns a single object
                    filings[company] = {tenks.filing_date: tenks.sections()}
                else:
                    filings[company] = {tenks[i].filing_date: tenks[i].sections() for i in range(self.years_back)}
                
                for section in self.section_patterns:
                        

            except Exception as e:
                print("Error retrieving filings for company: ", company)
        return filings


if __name__ == '__main__':
    er = EdgarRetriever(["AAPL"], 1, "Neal Lockhart neal301@gmail.com")