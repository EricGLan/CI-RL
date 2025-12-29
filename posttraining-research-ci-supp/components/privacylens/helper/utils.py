from azure.identity import ChainedTokenCredential, AzureCliCredential, DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI

def azure_api(messages, endpoint, temperature=0, max_tokens=5):
            
    credential = ChainedTokenCredential(
        AzureCliCredential(),
        DefaultAzureCredential(),
    )

    token_provider = get_bearer_token_provider(
        credential, "https://cognitiveservices.azure.com/.default"
    )
    client = AzureOpenAI(
        azure_endpoint=endpoint["endpoint"],
        azure_ad_token_provider=token_provider,
        api_version=endpoint["api_version"],
    )
    response = client.chat.completions.create(
        model=endpoint["deployment"],
        messages=messages,
        max_completion_tokens = max_tokens, 
        temperature = temperature
    )
    output_text = response.choices[0].message.content
        
    return output_text



