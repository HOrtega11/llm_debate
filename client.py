
import os
from openai import OpenAI


client = OpenAI(
    base_url="http://149.165.173.247:8888/v1",
    api_key=os.getenv("MY_API_KEY1")
)




