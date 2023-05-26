from tqdm import tqdm
import pymongo

class docn_mongo:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def retrieve(self, doccano_filename, retriever_filename, doccano_projectname):
        import sqlite3, json

        try:
            sqliteConnection = sqlite3.connect(doccano_filename)
            cursor = sqliteConnection.cursor()
            print("Successfully Connected to SQLite")

            # sqlite_select_Query = "select sqlite_version();"
            # cursor.execute(sqlite_select_Query)
            # record = cursor.fetchall()
            # print("SQLite Database Version is: ", record)

            sqlfile = open(retriever_filename)
            sql_string = sqlfile.read()
            sqlfile.close()

            cursor.execute(sql_string, {"project_name":doccano_projectname})

            columns = [_[0] for _ in cursor.description]
            data = cursor.fetchall()
            #print(data)
            cursor.close()

            results = [dict(zip(columns, _)) for _ in data]

            import pandas as pd
            df = pd.DataFrame(results)

            df_null=df[df["rel"].isna()] #entity pairs w/ no relations
            df_not_null = df[~df["rel"].isna()] #entity pairs w/ relations

            """the operation above is covered in SQL by:
            GROUP BY spans_id
            # df_null.drop_duplicates(keys, keep='first')"""

            df_null_original=df_null.loc[~df['spans_id'].isin(df_not_null["spans_id"])] #NULL entity pairs that do not overlap the spans with relations
            df = pd.concat([df_not_null,df_null_original]).sort_index() #concatenate results


            #SQLite does not support user-defined functions ->  implemented the entities order arrangement on the Python connector level
            def entity_order(s):
                indexes=["e1_id","e1_e","e1_start","e1_end","e2_id","e2_e","e2_start","e2_end"]

                return pd.Series([s["e1_id"],s["e1_e"],s["e1_start"],s["e1_end"],\
                   s["e2_id"],s["e2_e"],s["e2_start"],s["e2_end"]] , index=indexes) \
                       if s['e1_start'] < s['e2_start'] \
                       else pd.Series([s["e2_id"],s["e2_e"],s["e2_start"],s["e2_end"],
                                      s["e1_id"],s["e1_e"],s["e1_start"],s["e1_end"]],index=indexes)
            df[["e1_id","e1_e","e1_start","e1_end","e2_id","e2_e","e2_start","e2_end"]] \
                = df.apply(lambda x: entity_order(x) , axis=1)

            #unpack meta to dictionary
            string_key = "meta"
            df[string_key] = df[string_key].apply(json.loads)
            
            return_columns=['meta','rel','user_id',
                        'e1_e','e1_start','e1_end',
                        'e2_e','e2_start','e2_end']


            results=df[return_columns].to_dict('records')
            return results

        except sqlite3.Error as error:
            print("Error while connecting to sqlite", error)
        finally:
            if sqliteConnection:
                sqliteConnection.close()
                print("The SQLite connection is closed")
            
    def get_mongoDB_access(self,protocol , login , password , ip , port,  db_name , collection_name):
        import pymongo

        client_connect_str = "{protocol}://{login}:{password}@{ip}:{port}/".format(protocol=protocol, login=login, password=password,
                                                          ip=ip, port=port)

        try:
            client = pymongo.MongoClient(client_connect_str)
        except Exception as e:
            print(e)
            print('\033[31m' + "Invalid MongoDB Client Connection string" + '\033[0m')
            return None

        try:
            db = client[db_name]
        except Exception as e:
            print(e)
            print('\033[31m' + "Invalid MongoDB collection name" + '\033[0m')
            return None
        try:
            col = db[collection_name]
        except Exception as e:
            print(e)
            print('\033[31m' + "Invalid MongoDB document name" + '\033[0m')
            return None

        return col

    def get_mongoDB_collection_data(self,col , search_q , limit) :
        cursor = col.find() if search_q is None else col.find(search_q)
        db_collections = list(cursor)
        db_collections = db_collections if (limit is None) or (len(db_collections) <= limit) else db_collections[:limit]
        return db_collections

    def updating_to_mongoDB(self, mongDB_collection , update_data):
        import pymongo

        for f, d in tqdm(update_data,desc="updating to mongoDB..."):
            mongDB_collection.find_one_and_update(
                f,
                {
                    "$set": {
                        'label' : d
                    }
                }
            )

    def make_update_form(self,dict_list) :
        result = []
        meta_fields = ["tweet_id","author_id"]
        exlabel_fields = ["text" , "meta"]

        for dict_d in dict_list :
            dict_m = dict_d["meta"]
            be_tuple = [{key: value for key, value in dict_m.items() if key in meta_fields} ,
                        {key: value for key, value in dict_d.items() if key not in exlabel_fields}]
            be_tuple[0]["tweet_id"]=int(be_tuple[0]["tweet_id"])
            result.append(tuple(be_tuple))

        return result

    def get_DSE_dataframe(self,**kwargs):
        import pandas as pd
        """
        getting raw dataset dataframe
        :param kwargs:
        :return: raw DSE dataframe
        """

        df_name = 'DSE_dataframe' if "pklname" not in kwargs else kwargs.get("pklname")
        colnames = {"drug": 'Drug_label', "sentence": 'sent', "event": 'Suicid_label', "relation": 'Relation_label',
                    "id": 'ID', 'warning': 'warning', 'sui_keyword': 'sui_keyword', 'sui_classifier': 'sui_classifier',
                    "filter": 'IT_filtering', "author_id": "author_id"}
        doccanos_relation_nums = {"ADE": 1, "Means": 2, "Treatment": 3, "Misc": 9, "ADE-withdrawal": 11}
        label_entity_set = ["DRUG", "SE", "Neg-SE"]

        try:
            dfDS=pd.read_pickle(df_name+".pkl")
        except Exception as e:

            import re
            R, E, D, S = colnames.get("relation"), colnames.get("event"), colnames.get("drug"), colnames.get("sentence")

            datatypes={
                        R:int, E:str,
                        D:str, S:str,
                        'Didx_from':int,'Didx_to':int,
                        'Eidx_from':int,'Eidx_to':int
                       }
            """Import data from INF"""
            if "INF" in kwargs :

                dfDS = pd.read_excel(kwargs.get("INF")
                                 , usecols=[S, E, D, R, colnames.get("filter")] if "add_colnames" not in kwargs else [S, E, D, R, colnames.get("filter") ]+ kwargs.get("add_colnames"))
            else :
                """Import data from MongoDB"""
                if "monogDB_ccs" in kwargs and "monogDB_db" in kwargs and "monogDB_col" in kwargs :

                    client_connect_string = kwargs.get("monogDB_ccs")
                    database_name = kwargs.get("monogDB_db")
                    collections_name = kwargs.get("monogDB_col")

                    mongo = self.get_mongoDB_access(client_connect_str=client_connect_string,
                                               db_name=database_name,
                                               collection_name=collections_name)
                    mongo_col = self.get_mongoDB_collection_data(col=mongo)
                    dfDS = []
                    for d in mongo_col :
                        label_d = d["label"]

                        temp_dict = {
                            colnames["sentence"]: d["text"],
                            colnames["event"]: "",
                            colnames["drug"]: "",
                            colnames["relation"]: "",
                            colnames["warning"]: d["warnings"],
                            colnames['sui_keyword']: d['sui_keywords'],
                            colnames["author_id"]: d["author_id"],
                            colnames["sui_classifier"]: d["sui_classifier"],
                            "user_id": None,
                            'Didx_from' : 0 , 'Didx_to' : 0 , 'Eidx_from' : 0 , 'Eidx_to' : 0
                        }
                        for lb in label_d :
                            e = lb["e2"]  if lb["e1"] == "DRUG" else lb["e1"]
                            ds = lb["e1_start"] if lb["e1"] == "DRUG" else lb["e2_start"]
                            de = lb["e1_end"] if lb["e1"] == "DRUG" else lb["e2_end"]
                            es = lb["e2_start"] if lb["e1"] == "DRUG" else lb["e1_start"]
                            ee = lb["e2_end"] if lb["e1"] == "DRUG" else lb["e1_end"]


                            temp_dict[colnames["drug"]] = temp_dict[colnames["sentence"]][ds:de]
                            temp_dict[colnames["event"]] = e
                            temp_dict["Didx_from"] = ds
                            temp_dict["Didx_to"] = de
                            temp_dict["Eidx_from"] = es
                            temp_dict["Eidx_to"] = ee
                            temp_dict[colnames["relation"]] = doccanos_relation_nums[lb["rel"]]
                            temp_dict["user_id"] = lb["user_id"]

                            dfDS.append(temp_dict)
                    dfDS = pd.DataFrame(dfDS)
                else :
                    return False




            print(f"Total number of rows: {len(dfDS.index)}")

            """mandatory set edit"""
            # 2 filtering out unnecessary rows
            if "exclude" in kwargs and isinstance(kwargs.get("exclude"), dict) and "INF" in kwargs :
                exclude_colname = next(iter(kwargs.get("exclude").items()))[0]
                exclude_col = dfDS.loc[:, exclude_colname]  # takes only first column for filtering
                exclude_vals = next(iter(kwargs.get("exclude").items()))[1]
                for exval in exclude_vals:  # filtering out unesessary parts
                    dropindex = dfDS[dfDS[exclude_colname] == float(exval)].index
                    dfDS.drop(dropindex, inplace=True)
                print(f"After filtering out unnecessary rows : {len(dfDS.index)}")

            # 3 drop NaN labels
            if "keep_NA_labels" not in kwargs and not kwargs.get("keep_NA_labels"):
                dfDS.dropna(
                    subset=[R, E, D, S],
                    how='any', inplace=True)
                print(f"After filtering out N/A labels : {len(dfDS.index)}")

            # 4 drop sentences > 512 symbols (BERTs architecture limitation)
            if "truncate" in kwargs and kwargs.get("truncate"):
                dfDS = dfDS[dfDS[S].apply(lambda x: len(str(x)) <= kwargs.get("truncate"))]
                print(f"After sentence length restriction ({str(int(kwargs.get('truncate')))}) : {len(dfDS.index)}")

            # 5 drop sentences where drug or event is not mentioned in the sentence
            if "keep_foreign_labels" not in kwargs and not kwargs.get("keep_foreign_labels") and "INF" in kwargs:
                tempcol = 'tempcol'
                for entity_col in [colnames.get("drug"), colnames.get("event")]:
                    dfDS[tempcol] = dfDS.apply(lambda x: x[entity_col].lower() in x[S].lower(), axis=1)
                    dfDS = dfDS.loc[dfDS[tempcol] == True]
                    dfDS = dfDS.drop(tempcol, 1)
                print(f"After drop sentences where drug or event is not mentioned in the sentence : {len(dfDS.index)}")

            # 6 drop drug - event overlapping
            if "DE_overlapping" not in kwargs or not kwargs.get("DE_overlapping") :
                tempcol = "tempcol"
                dfDS[tempcol] = dfDS.apply(lambda x: True if x[D].lower().find(x[E].lower()) == -1 and x[E].lower().find(x[D].lower()) == -1 else False, axis=1)
                dfDS = dfDS.loc[dfDS[tempcol] == True]
                A = [tempcol, colnames.get("filter")] if "INF" in kwargs else [tempcol]
                dfDS = dfDS.drop(A, 1)
                print(f"After filtering out drug-event overlapping : {len(dfDS.index)}")

            dfDS[R] = dfDS[R].fillna(0.0).astype(int)

            # 7 multiple event or drug appearance distribution
            def getIdx(sentence, entity):
                try:
                    # entity = entity.replace("(", "").replace(")", "") #The brackets are metacharacters in regex (used to capture groups)
                    a=re.finditer(re.escape(entity), sentence, flags=re.IGNORECASE) #re.escape automatically escapes the metacharacters
                    # a = re.finditer('(?i)'+entity, sentence, flags=re.IGNORECASE) #alternative
                    return [_.span() for _ in a]
                except Exception as e:
                    # print(e)
                    # print(sentence, entity)
                    # print('\n')
                    return []

            for i in ["Didx", "Eidx"]:
                if "INF" in kwargs:
                    dfDS[i] = dfDS.apply(lambda x: getIdx(x[S], x[D if i == "Didx" else E]), axis=1)
                    if "labels_explode" not in kwargs or kwargs.get("labels_explode"):
                        dfDS = dfDS.explode(i)
                        dfDS[i + "_from"] = dfDS.apply(lambda x: x[i][0], axis=1)
                        dfDS[i + "_to"] = dfDS.apply(lambda x: x[i][1], axis=1)
                        dfDS.drop(labels=i, axis=1, inplace=True)

                    dfDS['count'] = dfDS.groupby(S)[S].transform('count')
                    dfDS = dfDS.sort_values('count')
                    dfDS = dfDS.drop(['count'], 1)

                #inserting tags
                if "labels_explode" not in kwargs or kwargs.get("labels_explode"):
                    def insert_tags(row):
                        S=colnames.get("sentence")
                        # print(row[S])
                        text = row[S][:min(row["Didx_from"],row["Eidx_from"])+(3*0)]+"<e1>"+row[S][min(row["Didx_from"],row["Eidx_from"])+(3*0):]
                        text = text[:min(row["Didx_to"],row["Eidx_to"])+(4*1)]+"</e1>"+text[min(row["Didx_to"],row["Eidx_to"])+(4*1):]
                        text = text[:max(row["Didx_from"],row["Eidx_from"])+(3*3)]+"<e2>"+text[max(row["Didx_from"],row["Eidx_from"])+(3*3):]
                        text = text[:max(row["Didx_to"],row["Eidx_to"])+(4*3)+1]+"</e2>"+text[max(row["Didx_to"],row["Eidx_to"])+(4*3)+1:]
                        return text

                    dfDS["tags"] = dfDS.apply(lambda x: insert_tags(x), axis=1)

                    # reassuring the type
                    dfDS = dfDS.astype(datatypes)

            if"labels_explode" not in kwargs or kwargs.get("labels_explode"):
                print(f"Multiple event or drug appearance distribution   : {len(dfDS.index)}")

            dfDS.reset_index(inplace=True, drop=True)
            dfDS = dfDS.shift()[1:]  # reindex

            """saving results"""
            dfDS.to_pickle(df_name+'.pkl')
            dfDS.to_excel(df_name+".xlsx")

        return dfDS

    def doccanolabels_to_mongoDB(self, **kwargs ):

        mongodb_protocol = kwargs.get("mongodb_protocol")
        mongodb_login = kwargs.get("mongodb_login")
        mongodb_pass = kwargs.get("mongodb_pass")
        mongodb_ip = kwargs.get("mongodb_ip")
        mongodb_port = kwargs.get("mongodb_port")
        mongodb_db = kwargs.get("mongodb_db")
        mongodb_col = kwargs.get("mongodb_col")
        doccano_filename = kwargs.get("doccano_filename")
        retriever_filename = kwargs.get("retriever_filename")
        doccano_projectname = kwargs.get("doccano_projectname")


        """get labels in doccanoDB"""
        labeling_data = self.retrieve(doccano_filename, retriever_filename, doccano_projectname)
        updating_data = self.make_update_form(labeling_data)

        """update label to mongoDB"""
        mongo_col = self.get_mongoDB_access(mongodb_protocol,mongodb_login,mongodb_pass,mongodb_ip,mongodb_port, mongodb_db , mongodb_col )
        self.updating_to_mongoDB(mongDB_collection=mongo_col, update_data=updating_data)




def main(**kwargs) :
    kwargs.pop("config_path")
    doccano_mongo = docn_mongo(**kwargs)

    doccano_mongo.doccanolabels_to_mongoDB(**kwargs)


    """check retrieve data """
    # import json
    # for i in labeling_data :
    #     print(json.dumps(i, indent = 4))
    #     print("\n\n")

    '''
        몽고 db -> dataframe 
        '''

    # import pandas as pd
    # pd.set_option('display.max_columns' , None)
    # pd.set_option('display.max_rows', None)
    #
    # # host => connection string
    # # mongo_col = get_mongoDB_access(client_connect_str="mongodb://iot:dkelab522@155.230.36.61:27017/", db_name="NBDMR",
    # #                                collection_name="doccano_test")
    # df = get_DSE_dataframe(monogDB_ccs="mongodb://iot:dkelab522@155.230.36.61:27017/",
    #                        monogDB_db="NBDMR",
    #                        monogDB_col="doccano_test")
    # print(df)
    #


if __name__ == "__main__" :
    import configparser as cparser
    import argparse
    from argparse import RawTextHelpFormatter
    import json

    parser = argparse.ArgumentParser(description="doccanoDB => mongoDB",
                                     formatter_class=RawTextHelpFormatter)

    parser.add_argument('--config_path', '-cnfgp', dest="config_path", type=str,
                        help="config file path\n ",
                        default="config.ini")

    crawling_options = vars(parser.parse_args())
    config_path = crawling_options["config_path"]

    # config file read start
    config = cparser.ConfigParser()
    config.read(config_path)

    # config 파일 값 읽기
    for section in config.sections():
        for item in config.items(section):
            if section in ["MongoDBSetting","DoccanoSetting"] or item[0] == "retriever_filename" :
                if item[1] in ["True", "False"]:
                    crawling_options[item[0]] = config.getboolean(section, item[0])
                elif item[1] == "None":
                    crawling_options[item[0]] = None
                elif item[1].isdigit():
                    crawling_options[item[0]] = int(item[1])
                elif len(item[1]) > 1 and item[1][0] == "[" and item[1][-1] == "]":
                    replace_single_quot = item[1].replace("\'", "\"")
                    crawling_options[item[0]] = json.loads(replace_single_quot)
                else:
                    crawling_options[item[0]] = item[1]
            else : pass

    main(**crawling_options)




