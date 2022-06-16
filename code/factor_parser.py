"""
This script contains parser for factor returns data from Kennet French website.
"""
import re
import calendar
import requests
import pandas as pd

from bs4 import BeautifulSoup
from zipfile import ZipFile
from io import BytesIO

base_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
datalib_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html"

# these pieces of website code trigger errors when parsing from K. French website
# they are mostly related to old dataset links that were not removed from site code
site_exceptions = ["4 Portfolios Formed on Size",
                   "16 Portfolios Formed on Size, Book-to-Market",
                   "Historical Benchmark Returns",
                   'href']


def response_handler(url: str,
                     timeout: int = 30,
                     **response_kwargs):
    """Web-request response handler to catch different types of errors"""
    try:
        response = requests.get(url, timeout=timeout, **response_kwargs)
        response.raise_for_status()
    except requests.Timeout:
        print("ошибка timeout, url:", url)
    except requests.HTTPError as err:
        code = err.response.status_code
        print(f"ошибка url: {url}, code: {code}")
    except requests.RequestException:
        print("ошибка скачивания url: ", url)
    else:
        return response


class FactorRequest:
    """
    Class to request historical factor data from K. French website
    """

    def __init__(self,
                 base_url: str = base_url,
                 data_page_url: str = datalib_url):
        self.base_url = base_url
        self.data_page_url = data_page_url
        self.table_links = None
        self.table_names = None

    @staticmethod
    def parse_date(x: str,
                   frequency: str = 'daily'):
        """
        Converts integer dates in string format from K. French tables to pandas dates.
        Possible frequency values are:'daily', 'monthly', 'yearly'
        """
        x = x.strip()
        if frequency == 'daily':
            year, month, day = x[0:4], x[4:6], x[6:]
        elif frequency == 'monthly':
            year, month = x[0:4], x[4:]
            day = calendar.monthrange(int(year), int(month))[1]
        # not impelemented for yearly frequency
        else:
            pass
        return pd.to_datetime(f"{year}-{month}-{day}", format="%Y-%m-%d")

    def request_tables_data(self, exceptions=site_exceptions):
        """
        Returns a dictionary:
        {'description of the data set (long name)': {
                      'description': url for description of the dataset loaded,
                      'csv': web link to load dataset in csv format,
                      'txt': web link to load dataset in txt format,
                      'dataframe': dataset itself in pd.DataFrame format if already downloaded
         'data_url': link to dataset from the webpage}

        exceptions: list of strings.
            A list of string patterns to ignore when searching for dataset links inside site code.
        """
        print(f"Parsing webpage {self.data_page_url} ...")

        response = response_handler(self.data_page_url)
        soup = BeautifulSoup(response.text, "html.parser")
        page_text = str(soup)
        splitted = [x.strip() for x in page_text.split('<br/>')]
        link_dict = {}
        # search pattern  for 'b' tags containing table descriptions
        r = re.compile("<%s>(.+?)</%s>" % ("b", "b"), re.I | re.S)

        print(f"Extracting links for tables with data...")

        for text in splitted:
            # find all text block that contain hrefs with zip archives
            if '.zip' in text:
                table_desc = re.findall(r, text)[0]
                # site author has not removed some old code with links and without description
                if all([x not in table_desc for x in exceptions]):
                    # get all links for zip archive associated with this table description
                    links = [link.get('href') for link in BeautifulSoup(text, "html.parser").find_all('a')]
                    link_dict[table_desc] = {
                        'csv': [x for x in links if 'CSV' in x][0],
                        'txt': [x for x in links if 'TXT' in x][0],
                        'description_link': [x for x in links if 'html' in x][0],
                        'dataframe': None
                    }

        self.table_links = link_dict
        self.table_names = list(link_dict.keys())

    def get_table(self,
                  table_name: str,
                  file_type: str = 'csv',
                  data_encoding: str = 'utf-8'
                  ) -> pd.DataFrame:
        """
        Returns Pandas dataframe with factor returns from Kennet French website

        Parameters:
        ----------
        table_name: str.
            Table name from K. French website ("Data Library" page). Example: "Fama/French 3 Factors".
        file_type: str, default 'csv'.
            Which url for data file to return. K. French website contains both 'txt' and 'csv' files.
        """
        # get all tables and their links
        if not self.table_links:
            self.request_tables_data()
        # if dataframe has been loaded already, return it immediately
        if self.table_links[table_name]['dataframe']:
            return self.table_links[table_name]['dataframe']

        try:
            data_url = self.table_links[table_name][file_type]
        except KeyError as e:
            print("There is no dataset with the name you supplied. Try again", e)

        response = response_handler(f"{self.base_url}{data_url}",
                                    timeout=30,
                                    stream=True)

        with ZipFile(BytesIO(response.content)) as z:
            for f in z.namelist():
                # read all data from file inside archive
                lines = [l.decode(data_encoding).strip().split(',') for l in z.open(f).readlines()]
                # find first row which starts with digit (dates are saved as digits in Kennet French datasets)
                try:
                    first_int = next(n for n, v in enumerate(lines) if v[0].strip().isdigit())
                except StopIteration:
                    raise StopIteration(
                        "There is no integer values in first column of table." /
                        "Unable find beginning of parsed table and read data."
                    )

                table = pd.read_csv(z.open(f),
                                    skiprows=first_int - 1,
                                    parse_dates=[0],
                                    index_col=0,
                                    date_parser=lambda x: FactorRequest.parse_date(x)
                                    )

                table.index.name = 'date'
                table.columns = [x.lower().replace('-', '_')
                                 for x in table.columns]

                return table / 100
