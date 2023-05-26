# nbdmrcrawler 1.0.0 v

'''

Crawler

'''


class NBDMRCrawler:
    '''
    crawler super class
    '''

    def __init__(self, keyword=None, **kwargs):
        import pandas as pd

        for k, v in kwargs.items():  # kwargs로 받은 elements를 class attribute으로 설정
            setattr(self, k, v)

        self.queries = []
        if hasattr(self, 'restore'):  # restore attribute이 있으면 기존의 저장된 object를 사용
            import json
            from tqdm import tqdm
            print(f"Restoring from excel {self.restore}")
            self.papers = pd.read_excel(self.restore, sheet_name="result", index_col=None).to_dict('records')
            for _ in tqdm(self.papers, desc="Papers"):  # Papers에 저장된 object 데이터로 update
                _.update({"drugs": json.loads(_.get("drugs").replace("\'", "\""))})
                _.update({"PMIDa": _.get("PMID").split(",")})
                for attr in ["meshs", "filters"]:
                    if hasattr(_, attr):
                        del _[attr]
                    else : pass
            self.queries = pd.read_excel(self.restore, sheet_name="queries", index_col=None).to_dict('records')

        else:
            self.results = []  # results : 크롤링한 결과 저장
            self.tweets = []  # tweet : results의 값을 post_processing , split_to_sents 한 결과 저장 in tweet crawler
            self.papers = []  # papers : results의 값을 post_processing , split_to_sents 한 결과 저장 in pubmed crawler

    '''
    
    get drug & brand data 
    
    '''
    def from_excel(self, **kwargs):
        '''
        excel 파일에서 drugs 이름을 가져와 list로 반환하는 method
            path : excel file path

            columns : column list to handling in excel

        :param kwargs:  (path , columns)
        :return: type(list) - sorted drugs list
        '''
        import pandas as pd
        import re

        excp_dict = [
            # "immune globulin",
            # "immune",
            # "crotalidae immune f(ab')2",
            # "factor",
            "other",
            "",
            " ",
            "oil"
        ]

        path = kwargs.get("path")  # "drug_mapping_v3_210726_2.xlsx"
        drugs = pd.read_excel(path)
        columns = kwargs.get("columns")

        flat_list = []
        if columns is None:
            for sublist in drugs.values.tolist():
                for item in sublist:
                    if type(item) is str and ("other" not in item):
                        flat_list.append(item.replace("\u3000", " ").strip())
                    else : pass
        else:
            print(f"Getting druglist from {path}, using columns: {' '.join(columns)}")
            for sublist in drugs.loc[:, columns].values.tolist():
                for item in sublist:
                    if type(item) is str and ("other" not in item):
                        flat_list.append(item.replace("\u3000", " ").strip())
                    else : pass

        # drop_exp = []
        # for item in flat_list:
        #     if "," in item:
        #         idx = item.find(",")
        #         item = item[:idx]
        #         drop_exp.append(item.strip())
        #
        #     if "(" in item:
        #         item = re.sub(pattern=pattern1,repl="",string=item)
        #         drop_exp.append(item.strip())
        #
        #     if "/" in item:
        #         temp = item.split("/")
        #         for i in range(len(temp)):
        #             drop_exp.append(temp[i].strip())
        #
        #     else:
        #         drop_exp.append(item.strip())

        # res = list(set(drop_exp))
        # for item in res:
        #     for excp in excp_dict:
        #         if excp in item:
        #             res.remove(item)
        #             break
        #     if item in replace_dict.keys():
        #         res.remove(item)
        #         res.append(replace_dict[item])

        return sorted(list(set(x for x in flat_list if x not in excp_dict)))
        # return sorted(list(set(res)))
        # return res.values.tolist()

    def from_csv(self, **kwargs):
        '''
        csv 파일에서 drugs 이름을 가져와 list로 반환하는 method
            path : excel file path

            columns : column list to handling in excel

        :param kwargs:  (path , columns)
        :return: type(list) - sorted drugs list
        '''
        import csv
        drugs = []
        with open(kwargs.get('path'), newline='') as inputfile:
            for row in csv.reader(inputfile):
                drugs.append(row[0])
        return drugs

    def get_product_name(self, druglist, type_var):
        '''
        druglist를 통해 해당 drug가 첨가된 product를 찾고 dict or list로 반환하는 method
            d : drug , p : product

            type_var == True :
                drug에 상관없이 모든 product를 인자로 하는 list

                ex) list = [ p1,p2,p3,p4 ... ]

            type_var == False :
                drug마다 분리하여 dict안 value를 해당 key(drug)의 product list로 하는 dict

                ex) dict = {d1 : [p1,p2 ...] , d2 : [p6,p7 ...]}


        :param druglist: type(list) - drug list
        :param type_var: type(Boolean) - True( list ) , False( dict )
        :return:  (type_var:True - type(list)) , (type_var:False - type(dict))
        '''

        import requests, json
        from tqdm import tqdm

        product_list = []
        product_dict = {}

        service1_prefix = "https://rxnav.nlm.nih.gov/REST/rxcui.json?name="  # findRxcuiByString (get RxCUI with drug name)
        service2_prefix = "https://rxnav.nlm.nih.gov/REST/brands.json?ingredientids="  # getMultiIngredBrand (get product name with RxCUI)

        for drug in tqdm(druglist, desc="Get drugs content_id list"):  # druglist 순회
            sub_list = []

            if type(drug) == str:
                url_id = service1_prefix + drug + "&search=0"

                content_id = requests.get(url_id)  # valid url , http response check
                if content_id.status_code == 200:  # requests response 200 : OK
                    content_id = content_id.content
                    dict_id = json.loads(content_id)  # change json to dictionary
                    if dict_id['idGroup'].get('rxnormId'):
                        id_list = dict_id['idGroup']['rxnormId']

                        for id in id_list:
                            url = service2_prefix + id
                            content_pn = requests.get(url)
                            if content_pn.status_code == 200:
                                content_pn = content_pn.content
                                dict_pn = json.loads(content_pn)  # change json to dictionary
                                if 'conceptProperties' in dict_pn['brandGroup'] and dict_pn['brandGroup'].get(
                                    'conceptProperties'):
                                    pn_list = dict_pn['brandGroup']['conceptProperties']
                                    for pn in pn_list:
                                        if pn.get('name'):
                                            sub_list.append(pn['name'])
                                    if type_var:
                                        product_list.extend(sub_list)
                                    else:
                                        product_dict[drug] = sub_list
                                else : pass
                            else : pass
                else : pass
            else : pass

        if type_var:
            return list(set(product_list))
        else:
            return product_dict

    def get_drugs_name(self):
        '''
        object 생성 시 drugs를 param으로 받았다면 받은 param을 drugs attribute setting,

        받지 않았다면 druglist_path를 통해 excel or csv file 읽어서 drugs attribute setting

        :return: None - setting drugs attribute
        '''
        if not hasattr(self, 'drugs') or self.drugs == []:  # drugs attribute 이 없다면 or drugs 가 비었다면
            # if not isinstance(self.drugs, list):
            drugs = []
            if self.druglist_path:
                if self.druglist_path.endswith('.xlsx') or self.druglist_path.endswith('.xls'):
                    drugs = self.from_excel(path=self.druglist_path, columns=self.columns)
                elif self.druglist_path.endswith('.csv'):
                    drugs = self.from_csv(path=self.druglist_path)
                else : pass
            else : pass
        else:
            if self.druglist_path:
                if self.druglist_path.endswith('.xlsx') or self.druglist_path.endswith('.xls'):
                    drugs = self.from_excel(path=self.druglist_path, columns=self.columns)
                elif self.druglist_path.endswith('.csv'):
                    drugs = self.from_csv(path=self.druglist_path)
                else : pass
                drugs.extend(self.drugs)
            else:
                drugs = self.drugs

        # 중복 제거 후 self.drugs update
        self.drugs = list(set(drugs))

    def get_drugs_name_class(self, druglist, type_var, level=3):  # level 1~5
        import requests
        from bs4 import BeautifulSoup
        from tqdm import tqdm
        import pandas as pd

        class_list = []
        class_dict = {}

        url_prefix = "https://www.kegg.jp/dbget-bin/www_bfind_sub?mode=bfind&max_hit=1000&locale=en&serv=kegg&dbkey=kegg&keywords="
        url_suffix = "&page=1"
        url2 = "http://rest.kegg.jp/get/"

        for drug in tqdm(druglist):
            if type(drug) == str:
                url = url_prefix + drug + url_suffix
                content_id = requests.get(url)

                if content_id.status_code == 200:
                    content_id = content_id.content
                    content_id = BeautifulSoup(content_id, "html.parser")

                    ids = content_id.find_all("a")
                    del (ids[0], ids[len(ids) - 1])

                    if len(ids) != 0:
                        for i in range(len(ids)):
                            ids[i] = str(ids[i])
                            idx = ids[i].find('>')
                            if ids[i][idx + 1] == 'D':
                                id = ids[i][idx + 1:idx + 7]

                                url = url2 + id

                                content_cls = requests.get(url)
                                if content_cls.status_code == 200:
                                    content_cls = str(content_cls.content)
                                    content_cls = content_cls[content_cls.find("BRITE"):content_cls.find("DBLINKS")]

                                    if content_cls:
                                        content2 = content_cls.split('\\n')
                                        atc = content2[level].split()[0]
                                        drug_class = content2[level].replace(atc, "").strip()

                                        if type_var:
                                            class_list.append(drug_class)
                                        else:
                                            class_dict[drug] = drug_class
                                break

        if type_var:
            return class_list
        else:
            return class_dict

    '''
    
    crawling data processing 
    
    '''
    def post_processing(self):
        '''
        크롤링 결과를 저장한 attribute data를 전처리하여 split_to_snets 처리를 할 attribute에 update하는 method

        :return: None - crawling result attribute update
        '''
        pass

    def split_to_sents(self, classify=False, keywords=False, titles=False, regex_sentsplit=False, med7=False):  # wasnt tested on multiple filters/meshs
        '''
        papers attribute의 abstract , title을 sentence로 나누고 분석하여 결과를 결과를 저장힐 attribute에 update하는 method

        :param classify: type(Boolean) -
        :param keywords: type(Boolean) -
        :param titles: type(Boolean) -
        :param regex_sentsplit: type(Boolean) -
        :param med7: type(Boolean) -
        :return: None - papers attribute update
        '''

        pass


    '''
    
    method using request/retry & HTTP GET response  
    
    '''
    def create_headers(self):
        '''
        인증을 위한 http header 정보를 가지는 dict를 반환하는 함수

        :return: type(dict) - http header
        '''
        pass

    def retry_requests(self, retries, backoff_factor, status_forcelist):
        '''
        retry request session을 반환하는 method

        :param retries: type(int) - retry time when HTTP Error occurred
        :param backoff_factor: type(float) - retry delay time - first time delay = 0.0 , next 0.3 , 0.6 , 0.9 ...
        :param status_forcelist: type(list) - HTTP error to retry
        :return: <class 'requests.sessions.Session'>
        '''
        import requests
        from requests.adapters import HTTPAdapter
        from requests.packages.urllib3.util.retry import Retry

        with requests.Session() as s:
            retry = Retry(total=retries, read=retries, connect=retries, backoff_factor=backoff_factor,
                          status_forcelist=status_forcelist)
            adapter = HTTPAdapter(max_retries=retry)
            s.mount('http://', adapter)
            s.mount('https://', adapter)
            return s

    def connect_to_endpoint_with_retry(self, headers, retries=3, backoff_factor=0.3, status_forcelist=[500, 502, 504, 429]):
        '''
        이전에 설정한 endpoint로 api 사용해 결과(dict)를 반환하는 method

        HTTP 429 response 발생시 빠른 재 시도를 위함

        connect_to_endpoint의 sleep을 사용시 정상적인 response에도 sleep하여 시간적으로 손실 발생

        설정한 에러 (status_forxcelist) 가 발생할때만 retry 함

        실질적인 데이터 수집 부분

        :param headers: type(dict) - http header
        :param retries: type(int) - retry time when HTTP Error occurred
        :param backoff_factor: type(float) - retry delay time - first time delay = 0.0 , next 0.3 , 0.6 , 0.9 ...
        :param status_forcelist: type(list) - HTTP errors to retry
        :return: type( dict ) - HTTP GET response
        '''
        try :
            if "lang:en" not in self.query_params["query"]:
                self.query_params["query"] = self.query_params["query"] + " lang:en"
            else : pass

            response = self.retry_requests(retries, backoff_factor, status_forcelist).get(self.search_url, headers=headers,
                                                                                          params=self.query_params)
            print("\t[ {} response : {} ]".format(self.query_params["query"], str(response)))
            # TODO exception 처리 수정
            if response.status_code != 200:
                print(Exception(response.status_code, response.text))
                return None
            else :
                return response.json()
        # TODO karina - 에러 발생 시 출력 & return None
        except Exception as e :
            self.print_with_color(f"query params : {self.query_params}", 9)
            self.print_with_color(f"search url : {self.search_url}", 9)
            self.print_with_color("exception : " + str(e) , 9)
            return None

    def connect_to_endpoint(self, headers):
        '''
        이전에 설정한 endpoint로 api 사용해 결과(dict)를 반환하는 method

        실질적인 데이터 수집 부분

        :param headers: type(dict) - http header
        :return: type( dict ) - HTTP GET response
        '''
        import requests


        if "lang:en" not in self.query_params["query"]:
            self.query_params["query"] = self.query_params["query"] + " lang:en"
        else : pass
        try :
            response = requests.request("GET", self.search_url, headers=headers, params=self.query_params)
            print("\t" + str(response))

        except Exception as e :
            self.print_with_color(str(e) , 9)
            pass

    def connect_to_endpoint_id(self, headers):
        '''
        이전에 설정한 endpoint로 api 사용해 결과(dict)를 반환하는 method

        실질적인 데이터 수집 부분

        id 정보를

        :param headers: type(dict) - http header
        :return: type( dict ) - HTTP GET response
        '''
        import requests
        response = requests.request("GET", self.search_url_from_id, headers=headers, params=self.query_params_id)
        # print(response.status_code)
        if response.status_code != 200:
            print(Exception(response.status_code, response.text))
            return None
        else :
            return response.json()


    '''
    
    method related to datetime module
    
    '''
    def check_valid_date(self, inputdate, name):
        '''
        inputdate가 유효한 날짜인지 검사하는 method

        ex) 2021/01/01 - 정상적인 날짜로 return True

        ex) 2021/13/01 - 13월은 없기 때문에 return False

        ex) 2021/02/30 - 2월에 30일은 없기 때문에 return False

        :param inputdate: type(str) - date string - ex) "2021/01/01"
        :param name: type(str) - string to print error message
        :return: type(Boolean) - check valid or invalid inputdate
        '''
        import datetime
        system_date_format = '%Y/%m/%d'
        try:
            input_date = datetime.datetime.strptime(inputdate, system_date_format)
            return True
        except ValueError:
            print("*** " + name + " : " + inputdate + " is invalid date! ***")
            return False

    def make_datetime(self):
        '''
        from_date , to_date attribute을 가지고 datetime obj를 만들어 튜플로 반환하는 method

        :return: type(tuple) - (type( class 'datetime.datetime' ) , type( class 'datetime.datetime' ))
        '''
        import datetime

        if self.check_valid_date(self.from_date, "from_date") and self.check_valid_date(self.to_date, "to_date"):
            import re

            startdate = list(
                map(int, (re.sub('[-=+,#/\?:^.@*\"※~ㆍ!』‘|\(\)\[\]`\'…》\”\“\’·]', ' ', self.from_date)).split()))
            enddate = list(
                map(int, (re.sub('[-=+,#/\?:^.@*\"※~ㆍ!』‘|\(\)\[\]`\'…》\”\“\’·]', ' ', self.to_date)).split()))

            from_date = datetime.date(startdate[0], startdate[1], startdate[2])
            to_date = datetime.date(enddate[0], enddate[1], enddate[2])
        else:
            from_date = datetime.date(2021, 1, 1)
            to_date = datetime.date(2022, 1, 1)
            print("from_date attribute change to " + str(self.from_date))
            print("to_date attribute change to " + str(self.to_date))

        return (from_date, to_date)


    '''
    
    method related to mongoDB
    
    '''
    def get_mongoDB_access(self , host, db_name, collection_name):
        import pymongo

        try:
            client = pymongo.MongoClient(host)
        except Exception as e:
            print(e)
            print('\033[31m' + "Invalid MongoDB Client Connection string" + '\033[0m')
            return None

        try:
            db = client[db_name]
        except Exception as e:
            print(e)
            print('\033[31m' + "Invalid MongoDB db name" + '\033[0m')
            return None
        try:
            col = db[collection_name]
        except Exception as e:
            print(e)
            print('\033[31m' + "Invalid MongoDB collection name" + '\033[0m')
            return None

        return col

    def update_to_MongoDB(self):
        '''
        저장할 몽고DB 경로를 설정한 후 데이터를 몽고DB에 업데이트하는 method
        '''

        host = self.mongodb_host
        collection = self.mongodb_col
        documnet = self.mongodb_doc
        print(f"\nupdate to MongoDB({host}{collection}/{documnet})")

        # 트윗 , 펍메드 나누기
        if self.source == "tweet":
            self.inserting(host, self.tweets, collection, documnet)
        elif self.source == "pubmed":
            self.inserting(host, self.papers, collection, documnet)
        else :
            pass

    def inserting(self, host, attr, collection_name, documents_name):
        '''
        몽고디비의 경로가 유효한지 파악하고 유효한 경로라면 attr parameter에 저장된 이름의 attribute에서 값을 가져와 저장하는 method

        :param host: type(string) - 주소와 비밀번호를 포함한 문자열입니다.
        :param attr: type(string) - 몽고디비로 업데이트 할 데이터를 가지는 attribute의 이름 입니다.
        :param collection_name: type(string) - 컬렉션의 이름 문자열입니다.
        :param documents_name: type(string) - 컬렉션 내부의 특정 도큐먼트 이름 문자열입니다.
        :return: None
        '''

        import pymongo
        from tqdm import tqdm
        import datetime

        mongoDB_col = self.get_mongoDB_access(host,collection_name,documents_name)

        if mongoDB_col :
            for data in tqdm(attr, desc="inserting {}{}/{} ...".format(host, collection_name, documents_name)):
                for k, v in data.items():
                    if isinstance(v, datetime.date):
                        data[k] = v.strftime("%Y-%m-%d")
                        continue
                    else : pass
                mongoDB_col.insert_one(data)
        else : pass

    def new_inserting(self, update_data , host, collection_name, documents_name ):
        '''
        몽고디비의 경로가 유효한지 파악하고 유효한 경로라면 attr parameter에 저장된 이름의 attribute에서 값을 가져와 저장하는 method

        :param host: type(string) - 주소와 비밀번호를 포함한 문자열입니다.
        :param attr: type(string) - 몽고디비로 업데이트 할 데이터를 가지는 attribute의 이름 입니다.
        :param collection_name: type(string) - 컬렉션의 이름 문자열입니다.
        :param documents_name: type(string) - 컬렉션 내부의 특정 도큐먼트 이름 문자열입니다.
        :return: None
        '''

        import pymongo
        import datetime

        mongoDB_col = self.get_mongoDB_access(host, collection_name, documents_name)
        update_data["created_at"] = update_data["created_at"].strftime("%Y-%m-%d")
        mongoDB_col.find({},{"_id":0 , "update_time":0})



        mongoDB_col.update_one()

    def delete_all_data(self, host, collection_name, documents_name):
        '''
        몽고DB의 특정 도큐먼트의 데이터를 모두 지우는 메서드입니다.

        테스트시 사용한 메서드로 실제 사용시 주의를 바랍니다.

        :param host: type(string) - 주소와 비밀번호를 포함한 문자열입니다.
        :param collection_name: type(string) - 컬렉션의 이름 문자열입니다.
        :param documents_name: type(string) - 컬렉션 내부의 특정 도큐먼트 이름 문자열입니다.
        :return: None
        '''
        import pymongo
        client = pymongo.MongoClient(host)
        db = client[collection_name]
        col = db[documents_name]
        col.delete_many({})


    '''
    
    make excel & txt file in mongoDB data 
    
    '''
    def export_txt_from_url(self , host, collection_name, documents_name, source=None, search_q=None, file_name=None, text_width=50):
        '''
        몽고DB의 데이터를 텍스트 파일로 저장하는 method입니다.

        @param host: type(string) - 주소와 비밀번호를 포함한 문자열입니다.
        @param collection_name: type(string) - 컬렉션의 이름 문자열입니다.
        @param documents_name: type(string) - 컬렉션 내부의 특정 도큐먼트 이름 문자열입니다.
        @param source: type(str) - 몽고디비 데이터의 출처입니다.
        @param search_q: type(dict) - 몽고디비의 특정 데이터를 찾는 쿼리문입니다.
        @param file_name: type(str) - 저장할 텍스트 파일의 이름입니다.
        @param text_width: type(int) - 텍스트를 write 시에 간격입니다. 초기값 50이면 50자를 쓰고 줄바꿈합니다.
        @return: None = write & save txt file
        '''
        pass

    def advance_cell_size(self , file_name):
        '''
        엑셀파일을 load하여 셀의 column과 row를 알맞게 조절한 후 수정된 엑셀파일을 저장하는 메서드

        :param file_name:  type(string) - 불러올 엑셀파일 이름입니다.
        :return: none - 엑셀파일을 수정 후 저장합니다.
        '''
        pass

    def export_excelfile(self, host, collection_name, documents_name, limit=None, search_q=None, file_name=None,sentences_keys=None):
        '''
        몽고DB에서 가져온 데이터를 엑셀파일로 만드는 메서드 입니다.

        :param host: type(string) - 주소와 비밀번호를 포함한 문자열입니다.
        :param collection_name: type(string) - 컬렉션의 이름 문자열입니다.
        :param documents_name: type(string) - 컬렉션 내부의 특정 도큐먼트 이름 문자열입니다.
        :param limit: type(int) - 엑셀파일에 담을 데이터의 수를 제한하는 정수입니다. 초기값 None은 제한이 없이 모든 데이터를 export합니다.
        :param search_Q: type(dict) - 딕셔너리 형태의 쿼리문 입니다. 초기값 None은 모든 데이터를 MongoDB에서 가져옵니다.
        :param file_name: type(string) - 저장할 엑셀파일 이름입니다. 초기값 None을 사용시 Tweet20220209_to_label.xlsx 과 같이 그날의 날짜에 맞춰서 파일이 생성됩니다.
        :param sentences_keys: type(list) - sentence sheet에 추가할 column들의 list 입니다. 사용자 입력은 받지 않습니다.
        :return: None - 엑셀 파일이 저장됩니다.
        '''
        pass


    '''

    make histogram file in crawling data 

    '''
    def make_histogram_data_and_StatisticsDict(self ):
        '''
        histogram을 만들 data들로 만든 sorted list와 통계적인 값들로 만든 dictionary로 된 tuple을 반환하는 method

        @return: type(typle(list , dict)) - return tuple composed of list of histogram data and statistics data dictionary.
        '''

        pass

    def make_crawling_inform(self):
        '''
        크롤링관련 저장할 데이터를 문자열로 반환하는 method

        @return: type(str) - related crawler data string
        '''
        pass

    def make_histogram(self, datalist, crawling_inform, statistics_data, file_name):
        '''
        히스트그램을 만드는 method

        @param datalist: type(list) - data to make a histogram
        @param crawling_inform: type(str) - string with crawler infromation
        @param statistics_data: type(dict) - dictionary with statistics data
        @param file_name: type(str) - file name to save histogram file
        @return: None - make & save histogram file ( .png )
        '''
        pass


    '''

    check argument valid 

    '''

    def check_from_date(self, from_date, first_check=False):
        from datetime import datetime

        if from_date == "":
            if first_check:
                self.print_with_color("*" * 50, 33)
                print()
            else : pass
            self.print_with_color("\t*** from_date is empty ***", 9)
            return False
        else : pass

        now = datetime.now()

        split_date = from_date.split('/')
        if len(split_date) != 3:
            self.print_with_color(f"\t*** {from_date} is invalid date! ***", 9)
            return False
        else : pass

        system_date_format = '%Y/%m/%d'

        try:
            date_check = datetime.strptime(from_date, system_date_format)
        except ValueError:
            self.print_with_color(f"\t*** {from_date} is invalid date! ***", 9)
            return False

        from_datetime = datetime.strptime(from_date, "%Y/%m/%d")

        if (from_datetime - now).days >= 0:
            self.print_with_color(f"\t*** from_date : {from_date} , now : {now.strftime('%Y/%m/%d')}***", 9)
            self.print_with_color(f"\t*** from_date should be a day earlier than today. ***", 9)
            return False
        else : pass

        return True

    def check_to_date(self , to_date, from_date, first_check=False):
        from datetime import datetime

        if to_date == "":
            if first_check:
                self.print_with_color("*" * 50, 33)
                print()
            else : pass
            self.print_with_color("\t*** to_date is empty ***", 9)
            return False
        else : pass

        now = datetime.now()

        split_date = to_date.split('/')

        # 사용자 입력이 형식에 맞지 않게 되었다면 다시시도 ex) 20220301 ,올바른 형식은 2022/03/01
        if len(split_date) != 3:
            self.print_with_color(f"\t*** {to_date} is invalid date! ***", 9)
            return False
        else : pass

        system_date_format = '%Y/%m/%d'

        # 입력된 to_date가 정상적인 날짜인지 확인
        try:
            date_check = datetime.strptime(to_date, system_date_format)
        except ValueError:
            self.print_with_color(f"\t*** {to_date} is invalid date! ***", 9)
            return False

        to_datetime = datetime.strptime(to_date, "%Y/%m/%d")
        from_datetime = datetime.strptime(from_date, "%Y/%m/%d")

        # 입력된 to_date가 현재 날짜 이후면 크롤링 할 수 없어서 다시시도
        if (to_datetime - from_datetime).days < 1:
            self.print_with_color(f"\t*** from_date is {from_date} ***", 9)
            self.print_with_color(f"\t*** {to_date} should be after {from_date}! ***", 9)
            return False
        else : pass

        # 입력된 to_date가 from_date 보다 이전 날짜라면 다시시도
        if (to_datetime - now).days >= 0:
            self.print_with_color(f"\t*** to_date : {to_date} , now : {now.strftime('%Y/%m/%d')}***", 9)
            self.print_with_color(f"\t*** to_date should be a day earlier than today. ***", 9)
            return False
        else : pass

        return True

    def check_druglist_path(self , druglist_path, first_check=False):

        # 빈 입력 값을 받았다면 엑셀에서 drug를 받지 않는다고 입력 종료
        if not druglist_path:
            return True
        else :
            pass

        import os.path
        # 입력받은 path에 파일이 없으면 다시시도
        if not os.path.isfile(druglist_path):
            if first_check:
                self.print_with_color("*" * 50, 33)
                print()
            else :
                pass
            self.print_with_color(f"\t*** there is no file in {druglist_path} ***", 9)
            return False
        else :
            pass

        return True


    '''
    user's prompt print with color   
    '''

    def print_with_color(self , text, tfs_color):
        print(f'\033[38;5;{tfs_color}m' + text + '\033[0m')

