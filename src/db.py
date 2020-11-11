import os
import pymysql
from flask import jsonify
from google.cloud import bigquery


def open_connection():
    connection = pymysql.connect(host='34.72.92.23',
                                user='root',
                                password='dbtest',
                                db='inference_db')
    return connection


def get_data():
    conn = open_connection()
    with conn.cursor() as cursor:
        result = cursor.execute('SELECT * FROM inference_data;')
        data = cursor.fetchall()
        if result > 0:
            got_data = jsonify(data)
        else:
            got_data = 'No data in DB'
    conn.close()
    return got_data


def add_data(data):
    conn = open_connection()
    with conn.cursor() as cursor:
        cursor.execute('INSERT INTO inference_data (longitude, latitude, age_of_house, annual_income, inland, near_ocean, near_bay, num_rooms, price_prediction) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)', (data["longitude"], data["latitude"], data["housing_median_age"], data["median_income"], data["ocean_proximity_INLAND"], data["ocean_proximity_NEAR BAY"], data["ocean_proximity_NEAR OCEAN"], data["rooms_per_household"], data["price_prediction"]))
        print("Successfully added data to db")
    conn.commit()
    conn.close()

