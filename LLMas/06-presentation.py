# Databricks notebook source
# MAGIC %md
# MAGIC This notebook is available at https://github.com/databricks-industry-solutions/review-summarisation.

# COMMAND ----------

# MAGIC %md
# MAGIC # Presentation
# MAGIC
# MAGIC We have used our model and our summarisation task is complete. As for the short and last step, the only thing that is left to be done is to turn our dataframe into an easily presentable format.
# MAGIC
# MAGIC What we want to aim for is a dataframe that has a row per book per week. In each row, we want to have some metadata information such as book name, author, etc.. as well as avg. rating for the week, positive summaries and negative summaries. This can greatly help if we need to build a dashboard.
# MAGIC
# MAGIC ----
# MAGIC
# MAGIC **Setup Used:**
# MAGIC
# MAGIC - Runtime: 13.2 ML
# MAGIC - Cluster:
# MAGIC   - Machine: 16 CPU + 64 GB RAM (For Driver & Worker)
# MAGIC   - 8 Workers

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data Defaults
# MAGIC Specify catalog and schema.

# COMMAND ----------

# Imports
from config import CATALOG_NAME, SCHEMA_NAME, USE_UC

# If UC is enabled
if USE_UC:
    _ = spark.sql(f"USE CATALOG {CATALOG_NAME};")

# Sets the standard database to be used in this notebook
_ = spark.sql(f"USE SCHEMA {SCHEMA_NAME};")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Read Data
# MAGIC Read the summarised and condensed dataframe.

# COMMAND ----------

# 4x total core count
spark.conf.set("spark.sql.shuffle.partitions", 512)

# Read the table
reviews_df = spark.read.table("book_reviews_condensed")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Build Meta DF
# MAGIC This dataframe will have the per week per book information.

# COMMAND ----------

# Imports
from pyspark.sql import functions as SF

# Build meta reviews df
meta_reviews_df = (
    reviews_df
    .withColumn(
        "weighted_star_rating", 
        SF.col("n_reviews") * SF.col("avg_star_rating")
    )
    .groupBy("asin", "title", "author", "week_start")
    .agg(
        SF.sum("n_reviews").alias("n_reviews"),
        SF.sum("n_review_tokens").alias("n_review_tokens"),
        SF.sum("weighted_star_rating").alias("weighted_star_rating"),
    )
    .withColumn(
        "avg_star_rating", 
        SF.round(SF.col("weighted_star_rating") / SF.col("n_reviews"), 2),
    )
    .drop("weighted_star_rating")
    .orderBy("asin", "title", "author", "week_start")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Build Summary Reviews
# MAGIC This dataframe will have positive and negative reviews placed in the same row rather than having separate rows for each. We will use a pivot function for this.

# COMMAND ----------

# Imports
from pyspark.sql import functions as SF

# Build meta reviews df
summary_reviews_df = (
    reviews_df.groupBy("asin", "title", "author", "week_start")
    .pivot("star_rating_class")
    .agg(SF.first("final_review_summary"))
    .withColumnRenamed("high", "positive_reviews_summary")
    .withColumnRenamed("low", "negative_reviews_summary")
    .orderBy("asin", "title", "author", "week_start")
)



# COMMAND ----------

# MAGIC %md
# MAGIC ### Join Dataframes
# MAGIC Join the two dataframes we just created

# COMMAND ----------

summary_df = meta_reviews_df.join(
    summary_reviews_df, 
    how="inner", 
    on=["asin", "title", "author", "week_start"]
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parse as HTML 
# MAGIC Parse the summary cells as HTML columns so we can display nicely on our dashboard.

# COMMAND ----------

# Imports
from pyspark.sql import functions as SF
import html

# Build a UDF to convert to HTML
@SF.udf("string")
def convert_to_html(text):
    html_content = ""
    try:
        # Escape any existing HTML characters
        escaped_string = html.escape(text)
        # Replace newline characters with HTML line breaks
        html_content = escaped_string.replace("\n", "<br>")
    except:
        pass
    return html_content

# Apply 
summary_df = (
    summary_df
    .withColumn("positive_reviews_summary", convert_to_html("positive_reviews_summary"))
    .withColumn("negative_reviews_summary", convert_to_html("negative_reviews_summary"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Build Display ID
# MAGIC We might have some occurrences where a book might have the same name with another one, therefore we want to create a unique display ID thats made from book's name, author's name, and the ID of the book.

# COMMAND ----------

# Imports
from pyspark.sql import functions as SF

# Build UDF 
@SF.udf("string")
def build_display_id(title, author, asin):
    display_id = f"{title} by {author} ({asin})"
    return display_id

# Apply
summary_df = summary_df.withColumn(
    "display_id", build_display_id(SF.col("title"), SF.col("author"), SF.col("asin"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save Finalised Dataframe
# MAGIC
# MAGIC And our final product is ready.. we can go ahead and save

# COMMAND ----------

(
    summary_df
    .write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("book_reviews_finalised")
)

# COMMAND ----------

display(summary_df.limit(5))