import os

from dotenv import load_dotenv
from neo4j import GraphDatabase
from pymongo import MongoClient

load_dotenv()


def connect_to_mongodb(database_name):
    connection = MongoClient(os.getenv("MONGO_DB_CONNECTION_STRING"))
    db = connection[database_name]
    return db


def connect_to_neo4j():
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_CONNECTION_STRING"),
        auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
    )
    database = driver.session(database="neo4j")
    return database
