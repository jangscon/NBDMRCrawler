# nbdmrcrawler

**nbdmrcrawler** is Python crawler to collect drug-related data from multiple sites.  
- Supports crawling in Pubmed and Twitter  
- Supports saving crawled data in MongoDB  
- Supports histogram creation including distribution of suicide clasifier and statistics data  

## directory & files

```bash
├── config
│   ├── config.ini
│   └── donedrugs.txt
├── doccano
│   └── ...
├── histogram
│   └── ...
├── labels
│   └── ...
├── NBDMRCrawler.py
├── TweetCrawler.py
├── README.md
└── LISENCE
``` 

## setting 

```bash 
in config.ini

< if using mongodb to store brand & drug name >
...
[brands_dict setting]
; brand_dict update term DEFAULT= 100 days
    brand_dict_term = <brand_dict term int num>
; brand_dict mongoDB host url
    mongodb_brand_host = <monogDB host url>
; brand_dict mongoDB database name
    mongodb_brand_db = <mongoDB database name>
; brand_dict mongoDB collection name
    mongodb_brand_col = <mongoDB collection name>
; mongoDB find regex character list
    brands_find_re_charlist = []
...


...
[MongoDBSetting]
; tolabel mongoDB host url
    mongodb_host = <monogDB host url>
; tolabel mongoDB database name
    mongodb_db = <mongoDB database name>
; tolabel mongoDB collection name
    mongodb_col = <mongoDB collection name>
; tolabel mongoDB column name list to add sentences sheet
    sentences_keys = ["sui_keywords" , "sui_clasifier"]
[PreprocessingSetting]
    trigger_warnings = ["tw //" , "tw//","trigger warning //" ,"trigger warning//"]
...

...
[DoccanoSetting]
;doccano base url
    doccano_url = <doccano base url>
;doccano username
    doccano_username = <doccano username>
;doccano password
    doccano_password = <doccano password>
;doccano project id
    doccano_pid = <doccano project id>
;Threshhold used to send from mongoDB to doccano
    doccano_threshold = 0.7

```

## run python file 

```bash 
 python3 TweetCrawler.py -sd [start_date] -ed [end_date] -drgs [drug_list]
 
 ex) start_date , end_date = 2006/07/16
 ex) drug_list = drg_A , drg_B , drg_C 
 
```