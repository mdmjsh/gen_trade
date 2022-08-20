import pandas as pd
from helpers import write_df_to_s3
import os


# Test S3 connection
df = pd.DataFrame([dict(a=1, b=2)])
write_df_to_s3(df, os.environ['BUCKET'], 'test_df.csv')