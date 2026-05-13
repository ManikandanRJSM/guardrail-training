import pandas as pd
from helpers.GetEnv import GetEnv
import os
from pyspark.sql import SparkSession, functions as F

def createSparkSession():
    spark = SparkSession.builder.getOrCreate()
    return spark


def Load(df):
    
    # df.write.mode('overwrite').format("csv").options(header='true').save(f"{_env['DATA_LAKE_PATH']}/guardrails_inputs/guardrails_inputs.csv")
    final_df = df.toPandas()
    final_df.to_csv(f"{_env['DATA_LAKE_PATH']}/guardrails_inputs/guardrails_inputs.csv")
    print('Done')

def extract(_env):
    # Extraction Part
    bad_df_raw = pd.read_parquet("hf://datasets/Mindgard/evaded-prompt-injection-and-jailbreak-samples/dataset.parquet")

    splits = {'train': 'data/train-00000-of-00001-88ba0162028a73fc.parquet', 'validation': 'data/validation-00000-of-00001-1deeef95c3248fe0.parquet'}
    good_df_raw = pd.read_parquet("hf://datasets/OpenAssistant/oasst2/" + splits["train"])[['text', 'lang', 'role']]

    Transform(bad_df_raw, good_df_raw)
    


def Transform(bad_df_raw, good_df_raw):
    bad_df_raw['original_sample'] = (
        bad_df_raw['original_sample']
        .str.replace('ChatGPT', 'tiki-taka-bot', regex=False)
        .str.replace('chatGPT', 'tiki-taka-bot', regex=False)
    )
    
     
    SparkSession = createSparkSession()
    b_df_raw = SparkSession.createDataFrame(bad_df_raw) \
        .na.replace("ChatGPT", "tiki-taka-bot", subset=["original_sample"]).na.replace("ChatGPT", "tiki-taka-bot", subset=["original_sample"]) \
        .withColumns({
        "label" : F.lit(1),
        "text" : F.col('original_sample')
    }).select('text', 'label')
    
    g_df_raw = SparkSession.createDataFrame(good_df_raw).filter(F.col('lang') == 'en').filter(F.col('role') == 'prompter').withColumns({
        "label" : F.lit(0)
    }).select('text', 'label')
    
    final_df = b_df_raw.union(g_df_raw).orderBy(F.rand())
    Load(final_df)

if __name__ == "__main__":
    

    _env = GetEnv.get_env_variables()
    os.environ["HF_TOKEN"] = _env['HF_TOKEN']
    extract(_env)