import sqlite3

domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital']#, 'police']
dbs = {}
for domain in domains:
    db = 'db/{}-dbase.db'.format(domain)
    
    conn = sqlite3.connect(db)
    c = conn.cursor()
    dbs[domain] = c

def queryResult(domain, turn):
    """Returns the list of entities for a given domain
    based on the annotation of the belief state"""
    # query the db
    sql_query = "select * from {}".format(domain)

    flag = True
    #print turn['metadata'][domain]['semi']
    for key, val in turn['metadata'][domain]['semi'].items():
        if val == "" or val == "dont care" or val == 'not mentioned' or val == "don't care" or val == "dontcare" or val == "do n't care":
            pass
        else:
            if flag:
                sql_query += " where "
                val2 = val.replace("'", "''")
                #val2 = normalize(val2)
                # change query for trains
                if key == 'leaveAt':
                    sql_query += r" " + key + " > " + r"'" + val2 + r"'"
                elif key == 'arriveBy':
                    sql_query += r" " + key + " < " + r"'" + val2 + r"'"
                else:
                    sql_query += r" " + key + "=" + r"'" + val2 + r"'"
                flag = False
            else:
                val2 = val.replace("'", "''")
                #val2 = normalize(val2)
                if key == 'leaveAt':
                    sql_query += r" and " + key + " > " + r"'" + val2 + r"'"
                elif key == 'arriveBy':
                    sql_query += r" and " + key + " < " + r"'" + val2 + r"'"
                else:
                    sql_query += r" and " + key + "=" + r"'" + val2 + r"'"

    #try:  # "select * from attraction  where name = 'queens college'"
    #print sql_query
    #print domain
    num_entities = len(dbs[domain].execute(sql_query).fetchall())

    return num_entities