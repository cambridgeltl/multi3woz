import sqlite3
import json

domains = ['restaurant', 'hotel', 'attraction', 'train', 'hospital']#, 'police']

domain_to_header = {'restaurant': ['id', 'address', 'area', 'food', 'introduction', 'name', 'phone', 'postcode', 'pricerange', 'signature', 'type'],
                    'hotel': ['id', 'address', 'area', 'internet', 'parking', 'single', 'double', 'family', 'name', 'phone', 'postcode', 'pricerange', 'takesbookings', 'stars', 'type'],
                    'attraction': ['id', 'address', 'area', 'entrance', 'name', 'phone', 'postcode', 'pricerange', 'openhours', 'type'],
                    'train': ['trainID', 'arriveBy', 'day', 'departure', 'destination', 'duration', 'leaveAt', 'price'],
                    'hospital': ['id', 'address', 'name', 'phone', 'post', 'department']}

for domain in domains:
    with open(f'db/{domain}_db.json', encoding="utf-8") as json_f:
        data = json.load(json_f)
        conn = sqlite3.connect(f'db/{domain}-dbase.db')
        header = domain_to_header[domain]
        field_string = ", ".join(header)
        conn.execute(f'CREATE TABLE {domain} ({field_string})')
        for item in data:
            values_string = ", ".join(['?' for i in header])
            value_tuple = tuple([item[i] if i in item else "" for i in header])
            sql_query = f"INSERT INTO {domain} ({field_string}) VALUES ({values_string})"
            print(sql_query)
            conn.execute(sql_query, value_tuple)
        conn.commit()
        conn.close()