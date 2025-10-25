from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model_name='claude-sonnet-4-5-20250929', timeout=None, stop=None)

result = model.invoke('give me some important parameters to learn for chatmodels in langchain framework. for example temperature, max_output_token etc. give me [parameter]: what it, what it does/mean')

print(type(result))
print(result)

print(result.content)