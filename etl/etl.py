import pandas as pd
from helpers.GetEnv import GetEnv
import os



if __name__ == "__main__":

    _env = GetEnv.get_env_variables()

    os.environ["HF_TOKEN"] = _env['HF_TOKEN']
    # Extraction Part
    df = pd.read_parquet("hf://datasets/Mindgard/evaded-prompt-injection-and-jailbreak-samples/dataset.parquet")

    # Transformation begin here
    df['original_sample'] = (
        df['original_sample']
        .str.replace('ChatGPT', 'tiki-taka-bot', regex=False)
        .str.replace('chatGPT', 'tiki-taka-bot', regex=False)
    )
    df['converstation_type'] = '1'

    # Loading part
    df.to_csv(f"{_env['DATA_LAKE_PATH']}/guardrails_inputs/guardrails_inputs.csv")
    print(df[['original_sample', 'converstation_type']])