import json
import sys
import time
import boto3
import requests
from datetime import datetime

from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.utils import getResolvedOptions
from awsglue.job import Job
from pyspark.sql import Row

args = getResolvedOptions(sys.argv, ["JOB_NAME"])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args["JOB_NAME"], args)

# ---------------- Constants ----------------
IMDB_API_HOST = "https://imdb236.p.rapidapi.com/api/imdb"
TMDB_API_HOST = "https://api.themoviedb.org/3"

IMDB_SECRET_ARN = (
    "arn:aws:secretsmanager:eu-north-1:013486663648:secret:"
    "events!connection/IMDB_API_CONNECTION/0c5b5d72-bfe0-40c3-b9c1-8f9e27caa807-8MWTDP"
)
TMDB_SECRET_ARN = (
    "arn:aws:secretsmanager:eu-north-1:013486663648:secret:"
    "events!connection/TMDB_API_CONNECTION/5788f53d-4460-45ed-90a8-088fb5c580ff-X79n8u"
)


def load_secret_json(secret_arn):
    client = boto3.client("secretsmanager")
    response = client.get_secret_value(SecretId=secret_arn)
    secret_value = response.get("SecretString")
    if not secret_value:
        raise ValueError(f"SecretString missing for {secret_arn}")
    return json.loads(secret_value)


imdb_secret = load_secret_json(IMDB_SECRET_ARN)
imdb_api_key = (
    imdb_secret.get("apiKey")
    or imdb_secret.get("X-RapidAPI-Key")
    or imdb_secret.get("api_key")
)
if not imdb_api_key:
    raise ValueError("IMDB API key missing from secret payload")

tmdb_secret = load_secret_json(TMDB_SECRET_ARN)
tmdb_token = (
    tmdb_secret.get("token")
    or tmdb_secret.get("authorization")
    or tmdb_secret.get("Authorization")
    or tmdb_secret.get("bearer")
)
if not tmdb_token:
    raise ValueError("TMDB token missing from secret payload")
tmdb_auth_header = (
    tmdb_token
    if tmdb_token.lower().startswith("bearer ")
    else f"Bearer {tmdb_token}"
)

IMDB_HEADERS = {
    "X-RapidAPI-Key": imdb_api_key,
    "X-RapidAPI-Host": "imdb236.p.rapidapi.com",
}

TMDB_HEADERS = {
    "Authorization": tmdb_auth_header,
    "accept": "application/json",
}

content_df = spark.read.json("s3://oruc-imdb-lake/raw/stg_contentIDs/data.json")
content_df.printSchema()

print("Reading input from S3...")
print("content_df count:", content_df.count())
content_df.show(truncate=False)


# ---------------- Helper Functions - API Calls ----------------
def fetch_imdb_content(session, imdb_id, retries=5):
    url = f"{IMDB_API_HOST}/{imdb_id}"
    for attempt in range(retries):
        try:
            response = session.get(url, headers=IMDB_HEADERS, timeout=20)
        except requests.RequestException:
            response = None
        if response is not None and response.status_code == 200:
            return response.json()
        time.sleep(1 + attempt * 0.5)
    return None


def fetch_tmdb_content(session, tmdb_id, retries=5):
    url = f"{TMDB_API_HOST}/{tmdb_id}"
    for attempt in range(retries):
        try:
            response = session.get(url, headers=TMDB_HEADERS, timeout=20)
        except requests.RequestException:
            response = None
        if response is not None and response.status_code == 200:
            return response.json()
        time.sleep(1 + attempt * 0.5)
    return None


def normalize_image(value):
    if isinstance(value, dict):
        return value.get("url")
    return value


def normalize_release_date(value):
    if isinstance(value, dict):
        return value.get("date") or value.get("year")
    return value


def enrich_partition(rows):
    session = requests.Session()
    now_ts = datetime.utcnow()
    for row in rows:
        imdb_id = row["id"]
        tmdb_id = row["tmdb_id"]
        content_type = row["type"]

        imdb_data = fetch_imdb_content(session, imdb_id)
        if not imdb_data:
            print("IMDB DATA EMPTY FOR:", imdb_id)
            continue
        tmdb_data = fetch_tmdb_content(session, tmdb_id)
        if not tmdb_data:
            print("TMDB DATA EMPTY FOR:", imdb_id)
            tmdb_data = {}

        yield (
            "detail",
            {
                "content_id": imdb_id,
                "content_type": content_type,
                "primary_title": imdb_data.get("primaryTitle"),
                "original_title": imdb_data.get("originalTitle"),
                "release_date": normalize_release_date(imdb_data.get("releaseDate")),
                "trailer": imdb_data.get("trailer"),
                "runtime_minutes": imdb_data.get("runtimeMinutes"),
                "content_poster": normalize_image(imdb_data.get("primaryImage")),
                "average_rating": imdb_data.get("averageRating"),
                "vote_count": imdb_data.get("numVotes"),
                "content_homepage": tmdb_data.get("homepage"),
                "overview": tmdb_data.get("overview"),
                "original_language": tmdb_data.get("original_language"),
                "status": tmdb_data.get("status"),
                "tagline": tmdb_data.get("tagline"),
                "budget": tmdb_data.get("budget"),
                "revenue": tmdb_data.get("revenue"),
                "created_at": now_ts,
                "updated_at": now_ts,
            },
        )

        for pc in tmdb_data.get("production_companies", []):
            yield (
                "production",
                {
                    "content_id": imdb_id,
                    "company_id": pc.get("id"),
                    "company_name": pc.get("name"),
                    "company_poster": (
                        f"https://image.tmdb.org/t/p/w92{pc.get('logo_path')}"
                        if pc.get("logo_path")
                        else None
                    ),
                },
            )

        for g in imdb_data.get("genres", []):
            yield ("genre", {"content_id": imdb_id, "genre_name": g})

        for interest in imdb_data.get("interests", []):
            yield ("interest", {"content_id": imdb_id, "interest_name": interest})

        for d in imdb_data.get("directors", []):
            yield (
                "person",
                {
                    "content_id": imdb_id,
                    "person_id": d.get("id"),
                    "person_name": d.get("fullName"),
                    "person_homepage": d.get("url"),
                    "person_poster": None,
                    "role_type": "director",
                    "character_names": None,
                    "order_no": None,
                },
            )

        for idx, c in enumerate(imdb_data.get("cast", [])):
            yield (
                "person",
                {
                    "content_id": imdb_id,
                    "person_id": c.get("id"),
                    "person_name": c.get("fullName"),
                    "person_homepage": c.get("url"),
                    "person_poster": c.get("primaryImage"),
                    "role_type": c.get("job"),
                    "character_names": c.get("characters") or [],
                    "order_no": idx + 1,
                },
            )

        for idx, cc in enumerate(tmdb_data.get("created_by", [])):
            yield (
                "person",
                {
                    "content_id": imdb_id,
                    "person_id": cc.get("id"),
                    "person_name": cc.get("name"),
                    "person_homepage": None,
                    "person_poster": None,
                    "role_type": "creator",
                    "character_names": None,
                    "order_no": idx + 1,
                },
            )

        for n in tmdb_data.get("networks", []):
            yield (
                "network",
                {
                    "content_id": imdb_id,
                    "network_id": n.get("id"),
                    "network_poster": (
                        f"https://image.tmdb.org/t/p/w92{n.get('logo_path')}"
                        if n.get("logo_path")
                        else None
                    ),
                    "network_name": n.get("name"),
                },
            )


# ---------------- Main Enrichment Loop ----------------
result_rdd = content_df.rdd.mapPartitions(enrich_partition).cache()

content_detail = result_rdd.filter(lambda item: item[0] == "detail").map(
    lambda item: Row(**item[1])
)
content_person = result_rdd.filter(lambda item: item[0] == "person").map(
    lambda item: Row(**item[1])
)
content_production = result_rdd.filter(lambda item: item[0] == "production").map(
    lambda item: Row(**item[1])
)
content_genre = result_rdd.filter(lambda item: item[0] == "genre").map(
    lambda item: Row(**item[1])
)
content_network = result_rdd.filter(lambda item: item[0] == "network").map(
    lambda item: Row(**item[1])
)
content_interest = result_rdd.filter(lambda item: item[0] == "interest").map(
    lambda item: Row(**item[1])
)

print("=== COLLECTOR COUNTS ===")
print("content_detail:", content_detail.count())
print("content_person:", content_person.count())
print("content_production:", content_production.count())
print("content_genre:", content_genre.count())
print("content_network:", content_network.count())
print("content_interest:", content_interest.count())

# ---------------- Spark DataFrame ==> S3 (STAGING) ----------------
print("=== S3 WRITE TEST ===")
spark.createDataFrame([Row(test="ok", ts=datetime.utcnow())]).write.mode(
    "overwrite"
).parquet("s3://oruc-imdb-lake/raw/_glue_test/")
print("S3 WRITE TEST DONE")

BUCKET = "s3://oruc-imdb-lake/raw/"

if not content_detail.isEmpty():
    df = spark.createDataFrame(content_detail)
    print("content_detail df count:", df.count())
    df.write.mode("append").parquet(f"{BUCKET}content_detail/")
    print("content_detail written")
if not content_person.isEmpty():
    df = spark.createDataFrame(content_person)
    print("content_person df count:", df.count())
    df.write.mode("append").parquet(f"{BUCKET}content_person/")
    print("content_person written")
if not content_production.isEmpty():
    df = spark.createDataFrame(content_production)
    print("content_production df count:", df.count())
    df.write.mode("append").parquet(f"{BUCKET}content_production/")
    print("content_production written")
if not content_genre.isEmpty():
    df = spark.createDataFrame(content_genre)
    print("content_genre df count:", df.count())
    df.write.mode("append").parquet(f"{BUCKET}content_genre/")
    print("content_genre written")
if not content_network.isEmpty():
    df = spark.createDataFrame(content_network)
    print("content_network df count:", df.count())
    df.write.mode("append").parquet(f"{BUCKET}content_network/")
    print("content_network written")
if not content_interest.isEmpty():
    df = spark.createDataFrame(content_interest)
    print("content_interest df count:", df.count())
    df.write.mode("append").parquet(f"{BUCKET}content_interest/")
    print("content_interest written")

job.commit()
