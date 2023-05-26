from tqdm import tqdm
import pymongo

def get_mongoDB_access( protocol, login, password, ip, port, db_name, collection_name):
    import pymongo
    client_connect_str = "{protocol}://{login}:{password}@{ip}:{port}/".format(protocol=protocol, login=login,
                                                                               password=password,
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

def get_mongoDB_data( mongo_col ) :
    A = mongo_col.aggregate([
        {"$group": {"_id": "$text", "count": {"$sum": 1}}},
        {"$match": {"_id": {"$ne": None}, "count": {"$gt": 1}}},
        {"$sort": {"count": -1}},
        {"$project": {"name": "$_id", "_id": 0}}
    ])
    print(list(A))

def delete_duplicate_data_in_mongoDB(mongoCol , dupleColumn) :
    '''

    :param mongoCol: mongodb collection
    :param dupleColumn: Name of column to deduplication
    :return: -
    '''
    # 몽고디비 내부 dupleColumn 으로 설정한 field 값이 중복인 것들을 찾음
    cursor = mongoCol.aggregate(
        [
            {"$group": {"_id": f"${dupleColumn}", "unique_ids": {"$addToSet": "$_id"}, "count": {"$sum": 1}}},
            {"$match": {"count": { "$gte": 2 }}}
        ]
    )

    response = []
    for doc in list(cursor):
        del doc["unique_ids"][0]
        for id in doc["unique_ids"]:
            response.append(id)

    mongoCol.delete_many({"_id": {"$in": response}})

    pass

def get_mongoCol_count(mongoCol) :
    return mongoCol.count_documents({})

if __name__ == "__main__" :
    # 몽고디비-NBDMR-doccano_test 테스트

    '''
        - 기존 loremDict = 50 개
        - 서로 다른 test Dict = 50 개

        - 기존의 collection의 수를 N이라고 하면 (기존 컬렉션에서 중복이 없다고 한다면) 
        - insert 후의 수는 N + 100 개 
        - 중복 제거하면 N + 51 개 
    '''

    # text 필드에서 중복을 확인
    dupleColumn="text"

    # 테스트로 몽고디비에 insert할 로렘입숨과 로렘입숨과 다른 텍스트
    loremIpsum = "Lorem ipsum dolor sit amet, consectetur adipisicing elit"

    # 몽고디비 - NBDMR - doccano_test collection에 access
    mongoCol = get_mongoDB_access("mongodb", "iot", "dkelab522", "155.230.36.61", "27017", "NBDMR", "doccano_test")

    # 1. 처음 몽고디비
    firstMongoColLen = get_mongoCol_count(mongoCol)

    lorem = {"name": "John", "text": loremIpsum}
    notlorem = [{"name": "John", "text": loremIpsum+"0"} , {"name": "John", "text": loremIpsum+"49"}]

    print(f"\nduplicate test data(num:50) = {lorem}")
    print(f"not duplicate test data(num:50) = {notlorem[0]} \n\t\t\t\t\t\t\t\t ~ {notlorem[1]}\n")
    print(f"1. mongoDB collection length (no insert test data) : {firstMongoColLen}")

    # insert할 딕셔너리의 수는 50개
    loremDictLen = 50
    # loremDict 50개를 insert
    loremlist = [{"name": "John"+str(i), "text": loremIpsum} for i in range(loremDictLen)]


    mongoCol.insert_many(loremlist)

    # loremIpsum 뒤에 숫자를 하나씩 더 추가해 서로 다른 중복을 가지지 않는 dict 50개를 insert
    mongoCol.insert_many([{"name": "John", "text": loremIpsum+str(i)} for i in range(loremDictLen)])

    # 2. test dict들 넣고 나서 몽고디비
    secondMongoColLen = get_mongoCol_count(mongoCol)
    print(f"2. mongoDB collection length (insert test data) : {secondMongoColLen}")

    # 몽고디비에서 중복 제거
    delete_duplicate_data_in_mongoDB(mongoCol=mongoCol , dupleColumn=dupleColumn)

    # 3. 중복제거 후 몽고디비
    thirdMongoColLen = get_mongoCol_count(mongoCol)
    print(f"3. mongoDB collection length (duplicated data remove) : {thirdMongoColLen}\n")

    if thirdMongoColLen - firstMongoColLen == 51 :
        print("delete_duplicate_data_in_mongoDB function is work!! :)")
    else :
        print("delete_duplicate_data_in_mongoDB function is not work :(")

    # 이전 테스트한 결과 제거
    mongoCol.delete_many({"name": "John"})
    mongoCol.delete_many({"text": loremIpsum})