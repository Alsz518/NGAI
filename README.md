# Network Generation AI (NGAI)

##  Installation

NGAI does not require installation, you should just clone the code and run locally.
```Powershell
clone https://github.com/Alsz518/NGAI.git
cd NGAI
```
At the same time, please install the required third-party libraries:

```Powershell
pip install -r requirements
```


## Configuration

### LLM Configuration

First, you need to configure OpenAI-Key. Please create a `./config.yaml` file in the root directory and add the following content to the file (please modify the content to your own settings):

```yaml
OPENAI_API_TYPE: 'azure' #'azure'  OR 'openai'
# for 'openai'
OPENAI_KEY: 'sk-xxxxxxxxxxx' # your openai key
# for 'azure'
AZURE_MODEL: 'XXXXX' # your deploment_model_name 
AZURE_API_BASE: https://xxxxxxxx.openai.azure.com/ # your deployment endpoint
AZURE_API_KEY: 'xxxxxx' # your deployment key
AZURE_API_VERSION: '2023-03-15-preview'
```

Here we recommend using ChatGPT-3.5 to run as LLM. If you want to use your own LLM, please refer to [LangChain-Large Language Models](https://python.langchain.com/docs/modules/model_io/models/) to define Your own LLM. In this case, please modify the following sections in `./DataProcessBot.py` and `./SimulationProcessBot.py` to configure your own LLM.

```Python
OPENAI_CONFIG = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
if OPENAI_CONFIG['OPENAI_API_TYPE'] == 'azure':
    os.environ["OPENAI_API_TYPE"] = OPENAI_CONFIG['OPENAI_API_TYPE']
    os.environ["OPENAI_API_VERSION"] = OPENAI_CONFIG['AZURE_API_VERSION']
    os.environ["OPENAI_API_BASE"] = OPENAI_CONFIG['AZURE_API_BASE']
    os.environ["OPENAI_API_KEY"] = OPENAI_CONFIG['AZURE_API_KEY']
    llm = AzureChatOpenAI(
        deployment_name=OPENAI_CONFIG['AZURE_MODEL'],
        temperature=0,
        max_tokens=1024,
        request_timeout=60
    )
elif OPENAI_CONFIG['OPENAI_API_TYPE'] == 'openai':
    os.environ["OPENAI_API_KEY"] = OPENAI_CONFIG['OPENAI_KEY']
    llm = ChatOpenAI(
        temperature=0,
        model_name='gpt-3.5-turbo-16k-0613',  # or any other model with 8k+ context
        max_tokens=1024,
        request_timeout=60
    )
```

Fine, now, you can run `./SimulationProcessBot.py`.

```Powershel
python ./SimulationProcessBot.py
```
## Contact 

If you have any questions or suggestions about this project, please send me an Issue or contact us by email: 21376373@buaa.edu.cn
