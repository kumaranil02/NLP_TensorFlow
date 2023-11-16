# Databricks notebook source
# MAGIC %md
# MAGIC This notebook is available at https://github.com/databricks-industry-solutions/review-summarisation.

# COMMAND ----------

# MAGIC %md
# MAGIC # Summarisation
# MAGIC
# MAGIC Our instructions are ready and the reviews are waiting to be summarised! We can now take the necessary steps to begin our inference (summirisation task).
# MAGIC
# MAGIC Before we do so, it might help to do a couple of things.. We want to optimise the speed of inference as much as possible (without trading off quality) and we also want to distribute our inference so we can scale properly. 
# MAGIC
# MAGIC In this notebook, we will cover the optimisations that can be done pre-summarisation, and how we can parallelize the work.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Setup Used:**
# MAGIC
# MAGIC - Runtime: 13.2 ML + GPU
# MAGIC - Cluster:
# MAGIC   - Machine: GPU with > 20GB (For Driver & Worker) 
# MAGIC   - 3+ Workers
# MAGIC   - Required GPUs: Nvidia A100 or A10

# COMMAND ----------

# MAGIC %md
# MAGIC #### Library Installation
# MAGIC
# MAGIC We can start by installing the libraries we are going to need for this work. As always, you can choose to specify these using the cluster's configuration page so that the cluster can auto spawn with these libraries installed. Another benefit - the libraries stay there even if you detach from the notebook (which won't be the case here..)

# COMMAND ----------

# Install libraries
%pip install -qq flash-attn
%pip install -qq xformers
%pip install -qq torch==2.0.1
%pip install -qq ctranslate2==3.17
%pip install -qq triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir_sm90#subdirectory=python

# Restart Python Kernel
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data Defaults
# MAGIC Specifying our data defaults for catalog and schema

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
# MAGIC #### Paths
# MAGIC Specifying the paths we are going to use in this notebook..

# COMMAND ----------

# Import the OS system to declare a ENV variable
from config import MAIN_STORAGE_PATH
import os

# Setting up the storage path (please edit this if you would like to store the data somewhere else)
main_storage_path = f"{MAIN_STORAGE_PATH}/model_store"

# Declaring as an Environment Variable 
os.environ["MAIN_STORAGE_PATH"] = main_storage_path

# Set local model paths
local_model_path = f"{main_storage_path}/mpt-7b-instruct"
local_tokenizer_path = f"{main_storage_path}/mpt-7b-tokenizer"
local_model_optimised_path = f"{main_storage_path}/mpt-7b-ct2"

# COMMAND ----------

# MAGIC %md
# MAGIC ### GPU Determination
# MAGIC
# MAGIC We want to determine whether we are using A10s or A100s. Depending on this, we are going to adjust our batch sizes.

# COMMAND ----------

import torch

# Get available GPU memory
total_mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2  # in MB

# Decide depending on memory
gpu_type = "small" if total_mem < 70000 else "large"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Retrieval
# MAGIC
# MAGIC We created the batched instructions dataset in the last notebook, which was produced after our prompt engineering tests. This dataset includes a `model_instruction` column, which has the text we are going to feed to the LLM with the instructions.

# COMMAND ----------

# Read the instructions dataframe
instructions_df = spark.read.table("batched_instructions")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Inferecene Optimisations
# MAGIC
# MAGIC Lets see if we can optimise the speed of our inference.. There is a library called `CTranslate2` which can take existing transformer like models, and optimise them for inference. This can help us greatly, and reduce the resources we may need to use.
# MAGIC
# MAGIC The library works by converting an existing transformer into a generator. Which essentially has the same properties, but with some added options. 
# MAGIC
# MAGIC This library offers quantisation as well.. Quantisation helps with making the model run with a smaller footprint on the GPU. However, it comes with a trade-off - the answer quality begins to drop as you quantise further. 
# MAGIC
# MAGIC But, for some cases it might still make sense. If you would like use a more performant quantisation, you can definitely lower it here.

# COMMAND ----------

# External Imports
from ctranslate2.converters import TransformersConverter

# Initiate the converter
if os.path.isdir(local_model_optimised_path):
    print("Optimised model exists")
else:
    mpt_7b_converter = TransformersConverter(
        model_name_or_path=local_model_path,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

    # Request conversion
    mpt_7b_converter.convert(
        output_dir=local_model_optimised_path,
        quantization="bfloat16"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Converted Model & Tokenizer
# MAGIC
# MAGIC Our model has been converted and is now ready to do be tested for inference. Let's load it up with the tokenizer, and see what we can do..

# COMMAND ----------

# External Imports
from transformers import AutoTokenizer
import ctranslate2
import os
import time

# Define the paths
local_tokenizer_path = f"{main_storage_path}/mpt-7b-tokenizer"
local_model_optimised_path = f"{main_storage_path}/mpt-7b-ct2"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path, padding_side="left")
tokenizer.pad_token_id = tokenizer.eos_token_id

# Load the model
mpt_optimised_model = ctranslate2.Generator(
    model_path=local_model_optimised_path,
    device="cuda",
    device_index=0,
    compute_type="bfloat16"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test Flow Build
# MAGIC
# MAGIC We can build a test flow to see how the model does, and experiment with some parameters. We especially want to focus on the batch size parameter here to find it's sweet spot.

# COMMAND ----------

def run_inference(requests, batch_size):
    
    # Create a return dict
    return_dict = {}

    # Time
    encoding_start = time.time()

    # Encode requests with tokenizer
    batch_tokens = [tokenizer.encode(x) for x in requests]
    batch_tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in batch_tokens]

    # Time
    return_dict["encoding_time"] = round(time.time() - encoding_start, 4)
    generation_start = time.time()

    # Generate results with the model
    batch_results = mpt_optimised_model.generate_batch(
        batch_tokens,
        max_batch_size=batch_size,
        max_length=150,
        include_prompt_in_result=False,
        sampling_temperature=0.1,
    )
    
    # Time
    return_dict["generation_time"] = round(time.time() - generation_start, 4)
    decoding_start = time.time()

    # Decode results with the tokenizer
    decoded_results = [tokenizer.decode(x.sequences_ids[0]) for x in batch_results]

    # Time
    return_dict["decoding_time"] = round(time.time() - decoding_start, 4)
    return_dict["total_time"] = round(time.time() - encoding_start, 4)

    # Prepare and Return
    return_dict["results"] = decoded_results
    return return_dict

# COMMAND ----------

# MAGIC %md
# MAGIC Retrieving few examples from our dataset here so we can do some tests

# COMMAND ----------

# Random sample examples
examples = (
    instructions_df
    .sample(False, 0.01, seed=42)
    .select("model_instruction")
    .limit(100)
    .collect()
)
examples = [x[0] for x in examples]

# COMMAND ----------

# MAGIC %md
# MAGIC The code below can help us with identifying the optimal spot for the batch size parameter

# COMMAND ----------

# Batch sizes to be tested
batch_size_test = [1, 5, 10]

if gpu_type == "large":
    batch_size_test += [15, 20, 25, 30, 35]

# Speed Test
for batch_test in batch_size_test:
    try:
        print("-"*15)
        print("Batch Size", batch_test)
        _result = run_inference(examples, batch_size=batch_test)
        print(_result["total_time"])
    except Exception as e:
        print(e)

# COMMAND ----------

# MAGIC %md
# MAGIC It looks like **20** is the number we are looking for if we have `A100`. 
# MAGIC **10** might be better suited for `A10`.
# MAGIC Let see how the results look like when we use this parameter

# COMMAND ----------

# Determine the ideal batch size
ideal_batch_size = 20 if gpu_type == "large" else 10

results = run_inference(examples, batch_size=ideal_batch_size)

for key in results.keys():
    if "time" in key:
        print(f"{key}: {results[key]}")

for _request, _response in zip(examples, results["results"]):
    print("-" * 15)
    print(_request)
    print()
    print(_response)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Distributed Inference
# MAGIC
# MAGIC Now that our test flow works and we have a batch size we can use, lets build a similar flow, but this time to be distributed across a cluster.
# MAGIC
# MAGIC In the code below, what we will try to do is to create the entire inference needed to process the instructions, and pack it all up in a `Pandas UDF`. By this way, we will get to execute the same flow in each worker.
# MAGIC
# MAGIC When this code us run as a UDF on our dataframe, each worker is going to get a copy of it alongside with a piece of data, and an identical copy of the model we have created with `CTranslate` will be loaded in each worker to process the data.
# MAGIC
# MAGIC This can be done thanks to the nature of the `Pandas UDF`, which differs from the regular `PySpark UDF`. Pandas UDFs get to process the data in chunks as Pandas Series, where as Spark UDFs process the data row by row.. So one gets called per row, and the other gets called per data chunk..

# COMMAND ----------

# External Imports
from pyspark.sql import functions as SF
import pandas as pd
from typing import Iterator

# Build Inference Function
@SF.pandas_udf("string", SF.PandasUDFType.SCALAR_ITER)
def run_distributed_inference(iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:

    # External Imports
    from transformers import AutoTokenizer
    import ctranslate2
    import torch

    # Define the paths
    local_tokenizer_path = f"{main_storage_path}/mpt-7b-tokenizer"
    local_model_optimised_path = f"{main_storage_path}/mpt-7b-ct2"

    # Understand GPU size
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2  # in MB

    # Decide depending on memory
    gpu_type = "small" if total_mem < 70000 else "large"

    # Params
    temperature = 0.1
    max_new_tokens = 150
    batch_size = 20 if gpu_type == "large" else 10
    repetition_penalty = 1.05
    top_k = 50
    top_p = 0.9

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load the model
    mpt_optimised_model = ctranslate2.Generator(
        model_path=local_model_optimised_path,
        device="cuda",
        device_index=0,
        compute_type="bfloat16"
    )

    for requests in iterator:

        # Encode requests with tokenizer
        batch_tokens = [tokenizer.encode(x) for x in requests.to_list()]
        batch_tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in batch_tokens]

        # Batch results
        batch_results = mpt_optimised_model.generate_batch(
            batch_tokens,
            max_batch_size=batch_size,
            max_length=max_new_tokens,
            include_prompt_in_result=False,
            sampling_temperature=temperature,
            sampling_topk=top_k,
            sampling_topp=top_p,
            repetition_penalty=repetition_penalty,
        )

        # Batch decode
        decoded_results = [tokenizer.decode(x.sequences_ids[0]) for x in batch_results]

        yield pd.Series(decoded_results)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Apply Inference on Dataframe
# MAGIC
# MAGIC Our inference function is ready to go, we can now map it to apply it to our dataframe.
# MAGIC
# MAGIC Over here, we want to set the number of repartition to the number of worker nodes we have in our cluster. For example, if we have 1 driver node and 3 worker nodes (as specified in the setup), then we want to set the repartition number to 3. We have some code which can automatically set the number of repartition to the number of worker nodes we have. However, this can be overridden if needed. (For example, for multiple gpu per worker nodes)

# COMMAND ----------

# Imports
from pyspark import SparkContext

# Auto get number of workers
sc = SparkContext.getOrCreate()

# Subtract 1 to exclude the driver
num_workers = len(sc._jsc.sc().statusTracker().getExecutorInfos()) - 1  

# Set the batch size for the Pandas UDF
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", num_workers*1000)

# Repartition
instructions_df = instructions_df.repartition(num_workers)

# Run Inference
instructions_df = (
    instructions_df
    .withColumn("llm_summary", run_distributed_inference(SF.col("model_instruction")))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save the Dataframe
# MAGIC
# MAGIC As for the final step of our notebook, we can go ahead and save our dataframe.

# COMMAND ----------

# Save
(
    instructions_df
    .write
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("book_reviews_summarised")
)