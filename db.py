#db.py
import os
import pymysql
from flask import jsonify

db_user = os.environ.get('CLOUD_SQL_USERNAME')
db_password = os.environ.get('CLOUD_SQL_PASSWORD')
db_name = os.environ.get('CLOUD_SQL_DATABASE_NAME')
db_connection_name = os.environ.get('CLOUD_SQL_CONNECTION_NAME')


def open_connection():
    unix_socket = '/cloudsql/{}'.format(db_connection_name)
    try:
        if os.environ.get('GAE_ENV') == 'standard':
            conn = pymysql.connect(user=db_user, password=db_password,
                                unix_socket=unix_socket, db=db_name,
                                cursorclass=pymysql.cursors.DictCursor
                                )
    except pymysql.MySQLError as e:
        print(e)

    return conn


def get_table():
    conn = open_connection()
    with conn.cursor() as cursor:
        result = cursor.execute('SELECT * FROM i_table;')
        inferences = cursor.fetchall()
        if result > 0:
            got_inferences = jsonify(inferences)
        else:
            got_inferences = 'No inferences in DB'
    conn.close()
    return got_inferences

def add_inference(inference):
    conn = open_connection()
    with conn.cursor() as cursor:
        cursor.execute('INSERT INTO i_table (longitude, latitude, house_age, income, inland, near_bay, near_ocean, rooms) VALUES(%s, %s, %s, %s, %s, %s, %s, %s)', (inference["longitude"], inference["latitude"], inference["house_age"], inference["income"], inference["inland"], inference["near_bay"], inference["near_ocean"], inference["rooms"]))
    conn.commit()
    conn.close()
