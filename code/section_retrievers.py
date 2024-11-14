from edgar import *
import time


class EdgarRetriever:

    def __init__(self, companies: List[str], years_back: int):
        self.companies = companies
        self.years_back = years_back
        self.sleep_time = 0.1
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

    def _get_filings(self):
        filings = {}

        for company in self.companies:
            try:
                time.sleep(self.sleep_time)

                print("Retrieving filings for company: ", company)

                tenks = Company(company).get_filings(form="10-K").latest(self.years_back)
                filings[company] = {tenks[i].date: tenks[i].sections() for i in range(len(tenks))}

                print("Retrieved ", len(tenks), " filings for company: ", company)

            except Exception as e:
                print("Error retrieving filings for company: ", company)
        return filings


if __name__ == '__main__':
    try:
        # Test parameters
        companies = ['AAPL', 'MSFT', 'GOOGL']
        years_back = 2
        
        # Create instance and retrieve filings
        print(f"Initializing retriever for {len(companies)} companies, looking back {years_back} years...")
        retriever = EdgarRetriever(companies, years_back)
        
        # Get filings
        filings = retriever._get_filings()
        
        # Print basic results
        print("\nRetrieval Summary:")
        for company in filings:
            print(f"{company}: {len(filings[company])} filings retrieved")
            for date in filings[company]:
                print(f"  - Filing date: {date}, Sections: {len(filings[company][date])}")
                
    except Exception as e:
        print(f"An error occurred: {str(e)}")